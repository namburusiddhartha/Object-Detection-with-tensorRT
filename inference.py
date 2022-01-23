import mmcv
import numpy as np
import torch
from mmdet.datasets.pipelines import Compose
from torch2trt_dynamic import TRTModule
import time 
import mmcv
import pycuda.driver as cuda_driver
import pycuda.autoinit

def init_detector(trt_model_path, device='cuda:0'):
#    #if isinstance(cfg, str):
#    #    cfg = mmcv.Config.fromfile(cfg)
    device_num = int(device.split(":")[-1])
    cuda_driver.Device(device_num).make_context()
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_model_path, map_location= device))
    return model_trt

#def init_detector(trt_model_path):
#    import time
#    tt= time.time()
#    model_trt = TRTModule()
#    model_trt.load_state_dict(torch.load(trt_model_path))
#    print("inti time", time.time()- tt)
#    return model_trt

class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
#         import pdb; pdb.set_trace()

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']

        else:
            results['filename'] = None
            results['ori_filename'] = None

        if torch.is_tensor(results["img"] ) == True :
            sh = results['img'].shape
            results['img'] = results['img'].permute(0, 3, 1, 2)[0]
            results['img_shape'] = sh[1:]
            results['ori_shape'] = sh[1:]
            results['img_fields'] = ['img']
            results['pad_shape'] = sh[1:]
            results['scale_factor'] = np.array([1,1,1,1],
                                    dtype=np.float32)
            results['keep_ratio'] = True
            results['resize'] = False

        else:
            img = results["img"]
            #img = mmcv.imread(results['img'])
            #results['img'] = img.astype(np.float32)
#             results['img'] = results['img'].transpose(2, 0, 1)
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            results['img_fields'] = ['img']
            results['pad_shape'] = img.shape
            results['scale_factor'] = np.array([1,1,1,1],
                                 dtype=np.float32)
            #results['keep_ratio'] = True
            results['resize'] = False


        return results



class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
#         import pdb; pdb.set_trace()

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']

        else:
            results['filename'] = None
            results['ori_filename'] = None
       
        if torch.is_tensor(results["img"] ) == True :
            sh = results['img'].shape
            results['img'] = results['img'].permute(0, 3, 1, 2)
            results['img_shape'] = sh[1:]
            results['ori_shape'] = sh[1:]
            results['img_fields'] = ['img']
            results['pad_shape'] = sh[1:]
            results['scale_factor'] = np.array([1,1,1,1],
                                    dtype=np.float32)
            results['keep_ratio'] = True
            results['resize'] = False

        else:
            img = results["img"]
            #img = mmcv.imread(results['img'])
            #results['img'] = img.astype(np.float32)
#             results['img'] = results['img'].transpose(2, 0, 1)
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            results['img_fields'] = ['img']
            results['pad_shape'] = img.shape
            results['scale_factor'] = np.array([1,1,1,1],
                                 dtype=np.float32)
            #results['keep_ratio'] = True
            results['resize'] = False
            

        return results

def get_keypoints(keypoint_pred, det_bboxes):
        heatmap_w = keypoint_pred.shape[3]
        heatmap_h = keypoint_pred.shape[2]

        num_preds, num_keypoints = keypoint_pred.shape[:2]

        scale_factor = 1.0

        bboxes = det_bboxes / scale_factor

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

        return xy_preds


def inference_detector(model, img, cfg, device):

    device = torch.device(device)
    cfg = mmcv.Config.fromfile(cfg)

    test_pipeline = [LoadImage()] + cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    tensor = data["img"][0]
    tensor = tensor.unsqueeze(0).to(device)
    print("preprocessingdone")
    with torch.no_grad():
        result = model(tensor)
    processed_keypoints = get_keypoints(result[4], result[1][0])
    result = list(result)
    result[4] = processed_keypoints
    return result



def batch_inference_detector(model, imgs, cfg, device):
    """Inference image(s) with the detector.
    Args:
        model (nn.Module): The loaded detector.
        imgs (list[ndarray]): loaded images.
    Returns:
        detection results directly.
    """
    num_imgs = len(imgs)
    
    # build the data pipeline
    tt = time.time()
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    print(time.time()-tt)
    data = []
    for img in imgs:
        d = dict(img=img)
        data.append(test_pipeline(d)["img"][0][0])
    data = torch.stack(data)
    tt = time.time()  
    with torch.no_grad():
        result = model(data)
        print(time.time() - tt)
    return result


def inference_detector_old(model, img, cfg, device):
    if isinstance(cfg, str):
        cfg = mmcv.Config.fromfile(cfg)

    device = torch.device(device)

    tm = time.time()
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)

    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    #print(test_pipeline)
    # prepare data

    data = test_pipeline(data)
    print("Preprocessing time1:", time.time() -tm)

    tm = time.time()
    tensor = data['img'][0]
    if isinstance(tensor, mmcv.parallel.DataContainer):
        tensor = tensor.data
    tensor = tensor.unsqueeze(0).to(device)
    img_metas = data['img_metas']
    scale_factor = img_metas[0].data['scale_factor']
    scale_factor = torch.tensor(scale_factor,
                                dtype=torch.float32,
                                device=device)
    print("Preprocessing time2:", time.time() -tm)
    #print(scale_factor)
    with torch.no_grad():
        torch.cuda.synchronize()
        tm = time.time()
        result = model(tensor)
        torch.cuda.synchronize()
        print("Forward pass time:", time.time() -tm)
        result = list(result)
        result[1] = result[1] / scale_factor

    return result
