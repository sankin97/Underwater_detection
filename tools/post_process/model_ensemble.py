import torch
import os
import mmcv
from mmdet.models import build_detector


def get_model(config, model_dir):
    model = build_detector(config.model, test_cfg=config.test_cfg)
    checkpoint = torch.load(model_dir)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=True)
    return model


def model_average(modelA, modelB, alpha):
    # modelB占比 alpha
    for A_param, B_param in zip(modelA.parameters(), modelB.parameters()):
        A_param.data = A_param.data * (1 - alpha) + alpha * B_param.data
    return modelA


if __name__ == "__main__":
    ###########################注意，此py文件没有更新batchnorm层，所以只有在mmdetection默认冻住BN情况下使用，如果训练时BN层被解冻，不应该使用此py　＃＃＃＃＃
    #########逻辑上会　score　会高一点不会太多，需要指定的参数是　[config_dir , epoch_indices ,  alpha]　　######################
    config_dir = 'configs/underwater/cas_x101/cascade_rcnn_x101_64x4d_fpn_1x_mixup_waterweeds.py'
    epoch_indices = [10, 11, 12]
    alpha = 0.7

    config = mmcv.Config.fromfile(config_dir)
    # work_dir = config.work_dir
    work_dir = "/root/data/hk/underwater-objection-detection/work_dirs/cas_x101_64x4d_fpn_htc_1x/04-09-13:30:26/"
    model_dir_list = [os.path.join(work_dir, 'epoch_{}.pth'.format(epoch)) for epoch in epoch_indices]

    model_ensemble = None
    for model_dir in model_dir_list:
        if model_ensemble is None:
            model_ensemble = get_model(config, model_dir)
        else:
            model_fusion = get_model(config, model_dir)
            model_emsemble = model_average(model_ensemble, model_fusion, alpha)

    checkpoint = torch.load(model_dir_list[-1])
    checkpoint['state_dict'] = model_ensemble.state_dict()
    save_dir = os.path.join(work_dir, 'epoch_ensemble_10_11_12_0.7.pth')
    torch.save(checkpoint, save_dir)