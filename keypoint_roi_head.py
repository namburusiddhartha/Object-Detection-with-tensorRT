import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.models.builder import build_wraper, register_wraper
from torch import nn


@register_wraper(
    'mmdet.models.roi_heads.mask_scoring_roi_head.MaskScoringRoIHead')
@register_wraper('mmdet.models.roi_heads.dynamic_roi_head.DynamicRoIHead')
@register_wraper('mmdet.models.roi_heads.keypoint_roi_head.KeypointRoIHead')
class KeypointRoIHeadWraper(nn.Module):

    def __init__(self, module, wrap_config={}):
        super(KeypointRoIHeadWraper, self).__init__()
        self.module = module
        self.wrap_config = wrap_config

        self.bbox_roi_extractor = build_wraper(module.bbox_roi_extractor)

        self.bbox_head = build_wraper(
            module.bbox_head, test_cfg=module.test_cfg)
        if module.with_shared_head:
            self.shared_head = module.shared_head
        else:
            self.shared_head = None

        self.init_keypoint_head(module.keypoint_roi_extractor, module.keypoint_head)
        # init mask if exist
        self.enable_mask = False
        if 'enable_mask' in wrap_config and wrap_config[
                'enable_mask'] and module.with_mask:
            self.enable_mask = True
            self.init_mask_head(module.mask_roi_extractor, module.mask_head)


        self.test_cfg = module.test_cfg

    def init_mask_head(self, mask_roi_extractor, mask_head):
        self.mask_roi_extractor = build_wraper(mask_roi_extractor)
        self.mask_head = build_wraper(mask_head, test_cfg=self.module.test_cfg)

    def init_keypoint_head(self, keypoint_roi_extractor, keypoint_head):
        self.keypoint_roi_extractor = build_wraper(keypoint_roi_extractor)
        self.keypoint_head = build_wraper(keypoint_head, test_cfg=self.module.test_cfg)  

    def _bbox_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.shared_head is not None:
            bbox_feats = self.shared_head(bbox_feats)
        # rcnn
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _mask_forward(self, x, rois):
        mask_feats = self.mask_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.shared_head is not None:
            mask_feats = self.shared_head(mask_feats)

        # mask forward
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def _keypoint_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            keypoint_feats = self.keypoint_roi_extractor(
                x[:self.keypoint_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                keypoint_feats = self.shared_head(keypoint_feats)
        else:
            assert bbox_feats is not None
            keypoint_feats = bbox_feats[pos_inds]

        keypoint_pred = self.keypoint_head(keypoint_feats)
        keypoint_results = dict(keypoint_pred=keypoint_pred, keypoint_feats=keypoint_feats)
        return keypoint_results


    def forward(self, feat, proposals, img_shape):
        batch_size = proposals.shape[0]
        num_proposals = proposals.shape[1]
        rois_pad = mm2trt_util.arange_by_input(proposals, 0).unsqueeze(1)
        rois_pad = rois_pad.repeat(1, num_proposals).view(-1, 1)
        proposals = proposals.view(-1, 4)
        rois = torch.cat([rois_pad, proposals], dim=1)

        # rcnn
        bbox_results = self._bbox_forward(feat, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        bbox_head_outputs = self.bbox_head.get_bboxes(rois, cls_score,
                                                      bbox_pred, img_shape,
                                                      batch_size,
                                                      num_proposals,
                                                      self.test_cfg)

        num_detections, det_boxes, det_scores, det_classes = bbox_head_outputs
        result = [num_detections, det_boxes, det_scores, det_classes]

        #Keypoint_processing
        num_keypoint_proposals = det_boxes.size(1)
        rois_pad = mm2trt_util.arange_by_input(det_boxes, 0).unsqueeze(1)
        rois_pad = rois_pad.repeat(1, num_keypoint_proposals).view(-1, 1)
        keypoint_proposals = det_boxes.view(-1, 4)
        keypoint_rois = torch.cat([rois_pad, keypoint_proposals], dim=1)
        keypoint_pred = self._keypoint_forward(feat, keypoint_rois)
        heatmap_w = keypoint_pred.shape[3]
        heatmap_h = keypoint_pred.shape[2]

        num_preds, num_keypoints = keypoint_pred.shape[:2]

        scale_factor = 1.0

        bboxes = det_boxes / scale_factor

        offset_x = bboxes[:, 0]
        offset_y = bboxes[:, 1]

        widths =  (bboxes[:, 2] - bboxes[:, 0]).clamp(min=1)
        heights = (bboxes[:, 3] - bboxes[:, 1]).clamp(min=1)

        width_corrections  = widths / heatmap_w
        height_corrections = heights / heatmap_h

        keypoints_idx = torch.arange(num_keypoints, device=keypoint_pred.device)
        xy_preds = torch.zeros((num_preds, num_keypoints, 4)).to(keypoint_pred.device)

        for i in range(num_preds):
            max_score, _ = keypoint_pred[i].view(num_keypoints, -1).max(1)
            max_score = max_score.view(num_keypoints, 1, 1)

            tmp_full_res = (keypoint_pred[i] - max_score).exp_()
            tmp_pool_res = (keypoint_pred[i] - max_score).exp_()
            roi_map_scores = tmp_full_res / tmp_pool_res.sum((1, 2), keepdim=True)
            pos = keypoint_pred[i].view(num_keypoints, -1).argmax(1)
            x_int = pos % heatmap_w
            y_int = (pos - x_int) // heatmap_w

            x = (x_int.float() + 0.5)*width_corrections[i]
            y = (y_int.float() + 0.5)*height_corrections[i]

            xy_preds[i, :, 0] = x + offset_x[i]
            xy_preds[i, :, 1] = y + offset_y[i]
            xy_preds[i, :, 2] = keypoint_pred[i][keypoints_idx, y_int, x_int]
            xy_preds[i, :, 3] = roi_map_scores[keypoints_idx, y_int, x_int]

        result += [xy_pred]

        if self.enable_mask:
            # mask roi input
            num_mask_proposals = det_boxes.size(1)
            rois_pad = mm2trt_util.arange_by_input(det_boxes, 0).unsqueeze(1)
            rois_pad = rois_pad.repeat(1, num_mask_proposals).view(-1, 1)
            mask_proposals = det_boxes.view(-1, 4)
            mask_rois = torch.cat([rois_pad, mask_proposals], dim=1)

            mask_results = self._mask_forward(feat, mask_rois)
            mask_pred = mask_results['mask_pred']

            mc, mh, mw = mask_pred.shape[1:]
            mask_pred = mask_pred.reshape(batch_size, -1, mc, mh, mw)
            if not self.module.mask_head.class_agnostic:
                det_index = det_classes.unsqueeze(-1).long()
                det_index = det_index + 1
                mask_pad = mask_pred[:, :, 0:1, ...] * 0
                mask_pred = torch.cat([mask_pad, mask_pred], dim=2)
                mask_pred = mm2trt_util.gather_topk(
                    mask_pred, dim=2, index=det_index)
                mask_pred = mask_pred.squeeze(2)

            result += [mask_pred]

        return result
