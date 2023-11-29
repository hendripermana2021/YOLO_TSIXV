# SPARK (Space Parking Analyzer with Real-time Knowledge)

## Project Description
Please describe your Startup Campus final project here. You may should your <b>model architecture</b> in JPEG or GIF.
SPARK (Space Parking Analyzer with Real-time Knowledge) adalah proyek yang bertujuan untuk mengembangkan sistem cerdas yang dapat mendeteksi dan menganalisis tempat parkir kosong. Dalam lingkungan perkotaan yang padat, mencari tempat parkir dapat menjadi tugas yang menantang dan memakan waktu. SPARK dirancang untuk memberikan solusi efisien dan real-time untuk menangani masalah ini dengan memanfaatkan teknologi visi komputer dan kecerdasan buatan. menggunakan YOLO sebagai modelling Object Detection yang akan di aplikasikan kedalam project aplikasi, namun ada beberapa modifikasi yang dilakukan didalam architecturenya, dengan tujuan mengembangkan model yang lebih efficient, presisi dan ringan.

<p align="left"><img width="800" src="utils/figures/Focus.png"></p>  

## Contributor
| Full Name | Affiliation | Email | LinkedIn | Role |
| --- | --- | --- | --- | --- |
| M. Haswin Anugrah Pratama | Startup Campus, AI Track | ... | [link](https://www.linkedin.com/in/haswinpratama/) | Supervisor |
| Muhammad Fathurrahman | Universitas Negeri Semarang | fathur.031207@gmail.com | [link](https://www.linkedin.com/in/muhammad-fathurrahman-3254781aa/) | Team Lead |
| Arya Gagasan | Universitas Negeri Jakarta | aryagagas56@gmail.com | [link](https://www.linkedin.com/in/aryagagas/) | Team Member |
| Fiya Niswatus Sholihah | Universitas Airlangga | fiyaniswatussholihah@gmail.com | [link](https://www.linkedin.com/in/fiya-niswatus-sholihah-89797a21a/) | Team Member |
| Laily Farkhah Adhimah | Universitas Amikom Purwokerto | lailyfarkhaha@gmail.com | [link](https://www.linkedin.com/in/laily-farkhah-adhimah-13b953257/) | Team Member |
| Hendri Permana Putra | STMIK Triguna Dharma | hendripermana60@gmail.com | [link](https://www.linkedin.com/in/hendri-permana-putra-7399b0131/) | Team Member |
| Muhammad Adib Ardianto | Universitas Muria Kudus | Adibardianto21@gmail.com | [link](https://www.linkedin.com/notifications/?filter=all) | Team Member |


## Setup
### Prerequisite Packages (Dependencies)
- pandas==2.1.0
- openai==0.28.0
- google-cloud-aiplatform==1.34.0
- google-cloud-bigquery==3.12.0
- matplotlib>=3.2.2
- numpy
- opencv-python
- Pillow>=7.1.2
- PyYAML>=5.3.1
- requests>=2.23.0
- scipy
- tqdm>=4.64.0
- protobuf<4.21.3  # https://github.com/ultralytics/yolov5/issues/8012
- seaborn>=0.11.0
- ipython  # interactive notebook
- psutil  # system utilization
- thop  # FLOPs computation
- streamlit
- wget
- ffmpeg-python
- streamlit_webrtc
- torch

### Environment
| | |
| --- | --- |
| CPU | HP 240 G7 Notebook intel i5, 8-core CPU |
| GPU | Intel(R) UHD Graphics |
| ROM | 512 GB |
| RAM | 8 GB |
| OS | Windows 11 |
| Another Environment | Google Colab for Train |

## Dataset
Describe your dataset information here. Provide a screenshot for some of your dataset samples (for example, if you're using CIFAR10 dataset, then show an image for each class).
Pengumpulan Dataset didapat dari hasil kolektif foto-foto yang diambil dari internet, video dari yt maupun hasil pengambilan foto dari kamera secara langsung, berikut salah satu image yang kami kumpulkan, yang berasal dari internet, maupun diambil secara langsung.

## Results
### Model Performance
Describe all results found in your final project experiments, including hyperparameters tuning and architecture modification performances. Put it into table format. Please show pictures (of model accuracy, loss, etc.) for more clarity.
<p align="left"><img width="800" src="utils/figures/Focus.png"></p>

#### 1. Metrics
Inform your model validation performances, as follows:
- For classification tasks, use **Precision and Recall**.
- For object detection tasks, use **Precision and Recall**. Additionaly, you may also use **Intersection over Union (IoU)**.
- For image retrieval tasks, use **Precision and Recall**.
- For optical character recognition (OCR) tasks, use **Word Error Rate (WER) and Character Error Rate (CER)**.
- For adversarial-based generative tasks, use **Peak Signal-to-Noise Ratio (PNSR)**. Additionally, for specific GAN tasks,
  - For single-image super resolution (SISR) tasks, use **Structural Similarity Index Measure (SSIM)**.
  - For conditional image-to-image translation tasks (e.g., Pix2Pix), use **Inception Score**.

Feel free to adjust the columns in the table below.

| model | epoch | learning_rate | batch_size | optimizer | val_loss | val_precision | val_recall | ... |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vit_b_16 | 1000 |  0.0001 | 32 | Adam | 0.093 | 88.34% | 84.15% | ... |
| vit_l_32 | 2500 | 0.00001 | 128 | SGD | 0.041 | 90.19% | 87.55% | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | 

#### 2. Ablation Study
Any improvements or modifications of your base model, should be summarized in this table. Feel free to adjust the columns in the table below.

| model | layer_A | layer_B | layer_C | ... | top1_acc | top5_acc |
| --- | --- | --- | --- | --- | --- | --- |
| vit_b_16 | Conv(3x3, 64) x2 | Conv(3x3, 512) x3 | Conv(1x1, 2048) x3 | ... | 77.43% | 80.08% |
| vit_b_16 | Conv(3x3, 32) x3 | Conv(3x3, 128) x3 | Conv(1x1, 1028) x2 | ... | 72.11% | 76.84% |
| ... | ... | ... | ... | ... | ... | ... |

#### 3. Training/Validation Curve
Insert an image regarding your training and evaluation performances (especially their losses). The aim is to assess whether your model is fit, overfit, or underfit.
 
### Testing
Show some implementations (demos) of this model. Show **at least 10 images** of how your model performs on the testing data.

### Deployment (Optional)
Describe and show how you deploy this project (e.g., using Streamlit or Flask), if any.

## Supporting Documents
### Presentation Deck
- Link: https://...

### Business Model Canvas
Provide a screenshot of your Business Model Canvas (BMC). Give some explanations, if necessary.

### Short Video
Provide a link to your short video, that should includes the project background and how it works.
- Link: https://...

## References
Provide all links that support this final project, i.e., papers, GitHub repositories, websites, etc.
- Link: [Official YOLOV5 repository](https://github.com/ultralytics/yolov5/)
- Link: [yolov5-improvements-and-evaluation, Roboflow](https://blog.roboflow.com/yolov5-improvements-and-evaluation/)
- Link: [Focus layer in YOLOV5]( https://github.com/ultralytics/yolov5/discussions/3181)
- Link: [CrossStagePartial Network](https://github.com/WongKinYiu/CrossStagePartialNetworkss)
- Link: [CSPNet: A new backbone that can enhance learning capability of cnn](https://arxiv.org/abs/1911.11929)- Link: [Path aggregation network for instance segmentation](https://arxiv.org/abs/1803.01534)
- Link: [Efficientnet-lite quantization](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html)
- Link: [YOLOv5 Training video from Texas Instruments](https://training.ti.com/process-efficient-object-detection-using-yolov5-and-tda4x-processors)

## Additional Comments
Provide your team's additional comments or final remarks for this project. For example,
1. ...
2. ...
3. ...

## How to Cite
If you find this project useful, we'd grateful if you cite this repository:
```
@article{
...
}
```

## License
For academic and non-commercial use only.

## Acknowledgement
This project entitled <b>"SPARK ()"</b> is supported and funded by Startup Campus Indonesia and Indonesian Ministry of Education and Culture through the "**Kampus Merdeka: Magang dan Studi Independen Bersertifikasi (MSIB)**" program.
