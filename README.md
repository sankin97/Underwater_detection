第一次打比赛还是很不容易，感谢队友[@milleniums](https://github.com/milleniums). 一起合作

## 主要提分点
1、Mixup
2、DCN与多尺度训练测试
3、global context ROI
4、旋转数据增强

## A榜验证消融实验结果
<table>
    <tr>
        <th>Backbone</th>
        <th>DCN</th>
        <th>MS</th>
        <th>Mixup</th>
        <th>RandomRotate90</th>
        <th>GC</th>
        <th>mAP</th>
    </tr>
    <tr>
        <th>ResNet50-FPN</th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th>baseline</th>
    </tr>
    <tr>
        <th>ResNeXt101-FPN</th>
        <th>&#10003;</th>
        <th>&#10003;</th>
        <th></th>
        <th></th>
        <th></th>
        <th>baseline+3.35%</th>
    </tr>
    <tr>
        <th>ResNeXt101-FPN</th>
        <th>&#10003;</th>
        <th>&#10003;</th>
        <th>&#10003;</th>
        <th></th>
        <th></th>
        <th>baseline+4.25%</th>
    </tr>  
    <tr>
        <th>ResNeXt101-FPN</th>
        <th>&#10003;</th>
        <th>&#10003;</th>
        <th></th>
        <th></th>
        <th>&#10003;</th>
        <th>baseline+4.36%</th>
    </tr>
    <tr>
        <th>ResNeXt101-FPN</th>
        <th>&#10003;</th>
        <th>&#10003;</th>
        <th></th>
        <th>&#10003;</th>
        <th>&#10003;</th>
        <th>baseline+4.54%</th>
    </tr>
     <tr>
        <th>ResNeXt101-FPN</th>
        <th>&#10003;</th>
        <th>&#10003;</th>
        <th></th>
        <th>&#10003;</th>
        <th>&#10003;</th>
        <th>baseline+4.69%</th>
    </tr>   
</table>

## 代码环境及依赖

+ OS: Ubuntu16.10
+ GPU: 2080Ti * 4
+ python: python3.7
+ nvidia 依赖:
   - cuda: 10.0.130
   - cudnn: 7.5.1
   - nvidia driver version: 430.14
+ deeplearning 框架: pytorch1.1.0
+ 其他依赖请参考requirement.txt

- **预训练模型下载**
  - 下载mmdetection官方开源的htc的[resnext 64x4d 预训练模型](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth)

## 依赖安装及编译
- **依赖安装编译**

   1. 创建并激活虚拟环境
        conda create -n underwater python=3.7 -y
        conda activate underwater

   2. 安装 pytorch
        conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch
        
   3. 安装其他依赖
        pip install cython && pip --no-cache-dir install -r requirements.txt
   
   4. 编译cuda op等：
        python setup.py develop
   

## 模型训练及预测
    
   - **训练**
        
        x101_64x4d (htc pretrained):
        
		chmod +x tools/dist_train.sh

        ./tools/dist_train.sh configs/underwater/cas_x101/cascade_rcnn_x101_64x4d_fpn_1x.py 4
        
        (上面的4是我的gpu数量，请自行修改,另外根据config对应文件目录进行修改)

   	2. 训练过程文件及最终权重文件均保存在config文件中指定的workdir目录中

   - **预测**
        
        x101_64x4d (htc pretrained):

        chmod +x tools/dist_test.sh

        ./tools/dist_test.sh configs/underwater/cas_x101/cascade_rcnn_x101_64x4d_fpn_1x.py workdirs/cas_x101_64x4d_fpn_htc_1x/latest.pth 4 --json_out results/cas_x101.json


    2. 预测结果文件会保存在 /results 目录下

    3. 转化mmd预测结果为提交csv格式文件：
       
       python tools/post_process/json2submit.py --test_json cas_x101.bbox.json --submit_file cas_x101.csv

       最终符合官方要求格式的提交文件cas_x101.csv 位于 submit目录下

## Reference
[Baseline @zhengye](https://github.com/zhengye1995/underwater-objection-detection).
