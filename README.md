# stomach

## Get start 

* train : python train_sigmoid.py --model_save_path "path" 

* test : python test.py 

* conver pth to onnx 


## Dataset 위치

210.115.46.234  

/data-1/team2/junghwan/stomach_mid_new_220808/

train, test

| 학습 데이터 | 개수 |
| --- | --- |
| E - 식도 | 1410 |
| S1 | 651 |
| S2 | 675 |
| S3 | 712 |
| S4 | 584 |
| S5 | 848 |
| S6 | 201 |
| D1 - 십이지장 |  402 |
| D2 - 십이지장 | 412 |
| C-겸자 | 70 |


| 테스트 데이터 | 개수 |
| --- | --- |
| E - 식도 | 328 |
| S1 | 94 |
| S2 | 155 |
| S3 | 104 |
| S4 | 99 |
| S5 | 135 |
| S6 | 13 |
| D1 - 십이지장 |  143 |
| D2 - 십이지장 | 301 |
| C-겸자 | 10 |


## 배포(최종) 파일

weight 파일(pth파일)

NAS 경로 :  /volume1/ziovision/1. personal/박정환/stomach_model/

* stomach_effiecintetnet-b0_bce.pth

* stomach_model.onnx  