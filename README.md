# label_tool

 ### 1. Introduction:
简单的目标检测标注工具，适用于连续图像标注场景，在原有标注功能的基础上，增加了前后帧匹配功能，点击下一帧时，自动根据前一帧的标注结果，匹配目标

 ### 2. Requirements:
 ```
  opencv-python
  pandas
  pillow
  matplotlib
  numpy
  scipy
  sklearn
  ```

 ### 3. Running demos：
 ```
 python main_rgbd.py
 ```
 
 ![image](https://user-images.githubusercontent.com/78141454/115691147-5a44ab80-a390-11eb-9124-b639a8c9dfbd.png)

快捷键：
```
  Delete：删除当前选中的标注框
  Baxkspace：清空当前帧标注
  +：（同时按shift和=）保存当前结果并加载下一帧已有标注（不做自动匹配）
  s：保存当前结果
  Right：下一帧（保存当前结果加载下一帧，并进行自动匹配）
  Left： 上一帧
  a: x1 – 1
  d: x1 + 1
  w: y1 – 1
  z: y1 + 1
  j: x2 – 1
  l: x2 + 1
  i: y2 – 1
  m: y2 + 1
 ```


 ### Credits:
 该工具在https://github.com/virajmavani/semi-auto-image-annotation-tool的基础上做的修改、优化。
