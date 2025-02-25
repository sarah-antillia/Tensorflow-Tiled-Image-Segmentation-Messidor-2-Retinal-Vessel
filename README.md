<h2>Tensorflow-Tiled-Image-Segmentation-Messidor-2-Retinal-Vessel (2025/02/26)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <a href="https://www.adcis.net/en/third-party/messidor2/"><b>Messidor-2</b></a>
 based on Pretrained HRF Retinal Vessel Model, which was trained by 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a <b>pre-augmented tiled dataset</b> <a href="https://drive.google.com/file/d/1bCbZRej3_aOaYuvXbv0vYnrPold3aXPf/view?usp=sharing">
Augmented-Tiled-HRF-ImageMask-Dataset.zip</a>, which was derived by us from the following dataset:<br><br>
<a href="https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip">
Download the whole dataset (~73 Mb)</a> in <a href="https://www5.cs.fau.de/research/data/fundus-images/"><b>High-Resolution Fundus (HRF) Image Database</b></a>.
<br>
<br>


Please see also our experiments:<br>
<li>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel</a> based on 
<a href="https://www5.cs.fau.de/research/data/fundus-images/">High-Resolution Fundus (HRF) Image Database</a>.
</li>
<li>
<a href="https://github.com/sarah-antillia/Tensorlfow-Tiled-Image-Segmentation-Pre-Augmented-DRIVE-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-DRIVE-Retinal-Vessel</a> based on 
<a href="https://drive.grand-challenge.org/">DRIVE: Digital Retinal Images for Vessel Extraction</a>
</li>
<li>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</a> based on 
<a href="https://cecas.clemson.edu/~ahoover/stare/">STructured Analysis of the Retina</a>.
<br>
</li>
<li>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel">
Tensorflow-Image-Segmentation-Retinal-Vessel</a> based on <a href="https://researchdata.kingston.ac.uk/96/">CHASE_DB1 dataset</a>.
</li>
<br>

<b>Experiment Strategies</b><br>
<br>
<b>1. LABELS (antillia ground truth) for Messidor-2 master IMAGES</b><br>
The Messidor-2 dataset contains no LABELS (ground truth) data. Therefore, we created our own master LABELS (ground truth) from the 
original Messidor-2 IMAGES by using Tiled Image Segmentation method and Pretrained-HRF-Retinal-Vessel 
UNet Model which was trained by 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a <b>pre-augmented tiled dataset</b> <a href="https://drive.google.com/file/d/1bCbZRej3_aOaYuvXbv0vYnrPold3aXPf/view?usp=sharing">
Augmented-Tiled-HRF-ImageMask-Dataset.zip</a>.
<br><br>
<b>2. Messidor-2 Dataset </b><br>
We splitted the Messidor-2-master IMAGES and LABELES into test, train and valid subsets. 
<br><br>
<b>3. Train Messidor-2 Segmentation Model </b><br>
We trained and validated a TensorFlow UNet model by using the <b>Messidor-2 train and valid subsets</b>
<br>
<br>
<b>4. Evaluate Messidor-2 Segmentation Model </b><br>
We evaluated the performance of the trained UNet model by using the <b>Messidor-2 test</b> dataset
 by computing the <b>bce_dice_loss</b> and <b>dice_coef</b>. </b>
<br>
<br>
<b>5. Tiled Inference   </b><br>
We applied our Tiled Image Segmentation method to infer the Retinal Vessel for the mini_test images 
of the original Messidor-2 IMAGES of 2240x1488 pixels.<br><br>

<hr>
<b>Actual Tiled Image Segmentation for Messidor-2 IMAGES of 2240x1488 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (antillia ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/images/20051020_44261_0100_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/masks/20051020_44261_0100_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test_output_tiled/20051020_44261_0100_PP.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/images/20051021_57862_0100_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/masks/20051021_57862_0100_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test_output_tiled/20051021_57862_0100_PP.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/images/20051212_36525_0400_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/masks/20051212_36525_0400_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test_output_tiled/20051212_36525_0400_PP.png" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this HRFSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
<h3>1.1 <b>High-Resolution Fundus (HRF) Image Database</b></h3>

The dataset used here has been taken from the dataset 
<a href="https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip">
Download the whole dataset (~73 Mb)</a> in <a href="https://www5.cs.fau.de/research/data/fundus-images/"><b>High-Resolution Fundus (HRF) Image Databaset</b></a>.
<br><br>
<b>Introduction</b><br>
This database has been established by a collaborative research group to support comparative studies on 
automatic segmentation algorithms on retinal fundus images. The database will be 
iteratively extended and the webpage will be improved.<br>
We would like to help researchers in the evaluation of segmentation algorithms. 
We encourage anyone working with segmentation algorithms who found our database useful to send us 
their evaluation results with a reference to a paper where it is described. This way we can extend our database of algorithms with the given results to keep it always up-to-date.
<br>
The database can be used freely for research purposes. We release it under Creative Commons 4.0 Attribution License. 
<br><br>
<b>Citation</b><br>
<a href="https://onlinelibrary.wiley.com/doi/10.1155/2013/154860">
<b>Robust Vessel Segmentation in Fundus Images</b></a>
<br>
Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.<br>
<b>
International Journal of Biomedical Imaging, vol. 2013, 2013</b>
<br>
<br>

<b>Licence</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons 4.0 Attribution License.</a>
<br>
<br>
<h3>

<h3>1.2 <b>Messidor-2 Dataset</b></h3>
We also used <a href="https://www.adcis.net/en/third-party/messidor2/">Messidor-2 Dataset</a>,<br><br>
MESSIDOR stands for <b>M</b>ethods to <b>E</b>valuate <b>S</b>egmentation and <b>I</b>ndexing <b>T</b>echniques in the field of 
<b>R</b>etinal <b>O</b>phthalmology (in French).<br>
<br>
<b>Messidor-2 Dataset</b><br>
The Messidor-2 dataset is a collection of Diabetic Retinopathy (DR) examinations, each consisting of two macula-centered eye fundus images (one per eye).
Part of the dataset (Messidor-Original) was kindly provided by the Messidor program partners. 
The remainder (Messidor-Extension) consists of never-before-published examinations from Brest University Hospital.
In the original Messidor dataset, some fundus images came in pairs, some others were single. 
Messidor-Original consists of all image pairs from the original Messidor dataset, that is 529 examinations (1058 images, 
saved in PNG format).
In order to populate Messidor-Extension, diabetic patients were recruited in the Ophthalmology department of 
Brest University Hospital (France) between October 16, 2009 and September 6, 2010. Eye fundi were imaged, 
without pharmacological dilation, using a Topcon TRC NW6 non-mydriatic fundus camera with a 45 degree field of 
view. Only macula-centered images were included in the dataset. <br>
Messidor-Extension contains 345 examinations (690 images, in JPG format).<br>
Overall, Messidor-2 contains 874 examinations (1748 images). 
The dataset comes with a spreadsheet containing image pairing. 
<b>It does not contain annotations such as a diabetic retinopathy “ground truth”.
 However, some third-parties proposed such annotations, but these are independent from the official Messidor-2 database,
  and therefore not handled by our services.</b><br>
  <br>
<b>Using the database</b><br>
Messidor-2 can be used, free of charge, for research and educational purposes. Copy, redistribution, 
and any unauthorized commercial use are prohibited. 
Any publication relying on this dataset must acknowledge the 
<a href="https://latim.univ-brest.fr/index.php?lang=en&lang=en">LaTIM laboratory</a> and the Messidor program partners.
Please include the following acknowledgmen.<br>

Kindly provided by the Messidor program partners (see https://www.adcis.net/en/third-party/messidor/).
<br>
Decencière et al..<br>
Feedback on a publicly distributed database: the Messidor database.<br>
Image Analysis & Stereology, v. 33, n. 3, p. 231-234, aug. 2014. ISSN 1854-5165.<br>
Available at: http://www.ias-iss.org/ojs/IAS/article/view/1155 or
http://dx.doi.org/10.5566/ias.1155.

M. D. Abràmoff, J. C. Folk, D. P. Han, J. D. Walker, D. F. Williams, S. R. Russell, P. Massin, B. Cochener,<br>
 P. Gain, L. Tang, M. Lamard, D. C. Moga, G. Quellec, and M. Niemeijer,<br>
Automated analysis of retinal images for detection of referable diabetic retinopathy.<br>
JAMA Ophthalmol, vol. 131, no. 3, Mar. 2013, p. 351–357.<br>
Available at: https://doi.org/10.1001/jamaophthalmol.2013.1743.<br>
<br>


<h3>2. Create LABELS for Messidor-2 IMAGES</h3>
<h3>2.1 Download Messidor-2 IMAGES Dataset</h3>
Please download <a href="https://www.adcis.net/en/third-party/messidor2/">Messidor-2 Dataset</a>,
and place IMAGES under Messidor-2-master folder as shown below.
<pre>
./projects
 └─generator
    └─Messidor-2-master
        └─IMAGES
</pre>


<h3>2.2 Download Augmented-Tiled-HRF-Pretrained-Model</h3>

Please download our <a href="https://drive.google.com/file/d/1xsfPzQ8srbKr8qXo-mDyUJrne4FRG5nw/view?usp=sharing">
Augmented-Tiled-HRF-Pretrained-Model.zip,</a> expand it and place <b>best_model.h5</b> under models folder as shown below. 
<pre>
./projects
 └─TensorflowSlightlyFlexibleUNet
    └─Augmented-Tiled-HRF
         └─models
             └─best_model.h5
</pre>
<h3>2.3 Run Tiled Inference method</h3>

Please move the folder "./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-HRF". and run the following bat file.
<pre>
5.tiled_infer-messidor.bat
</pre>
This will generate our own LABELS (ground truth) for the official Messidor-2 IMAGES by using the HRF-Pretrained-Model,
without any human experts.<br>
<pre>
./projects
 └─generator
    └─Messidor-2-master
        ├─IMAGES
        └─LABELS
</pre>

<h3>2.4 Split Messidor-2-master</h3>
Please move to "./projects/generator" folder and run the following Python script.<br>
<pre>
python spit_master.py
</pre>
,by which the following Messidor-2 dataset will be created.
<pre>
./dataset
└─Messidor-2
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
This is a 2240x1488 pixels images and their corresponding masks dataset.<br>
.<br>

<br>
<b>Messidor-2 Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/Messidor-2_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough 
to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained HRF TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Messidor-2 and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Enabled Batch Normalization.<br>
Defined a small <b>base_filters=16</b> and large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Dataset class</b><br>
Specified ImageMaskDataset class.
<pre>
[dataset]
datasetclass  = "ImageMaskDataset"
resize_interpolation = "cv2.INTER_LINEAR"
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Tiled Inference</b><br>
Used the original Messidor-2 IMAGES as a mini_test dataset for our inference images.
<pre>
[tiledinfer] 
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output_tiled"
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer      = False
epoch_change_infer_dir  = "./epoch_change_infer"
epoch_change_tiledinfer = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images       = 6
</pre>

By using this callback, on every epoch_change, the epoch change tiled inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_tiled_inference output at starting (1,2,3,4)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/asset/epoch_change_tiled_infer_at_start.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_tiled_inference output at ending (75,76,77,78)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/asset/epoch_change_tiled_infer_at_end.png" width="1024" height="auto"><br>
<br>
<br>

In this experiment, the training process was stopped at epoch 78 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/asset/train_console_output_at_epoch_78.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Messidor-2</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Messidor-2/test.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/asset/evaluate_console_output_at_epoch_78.png" width="720" height="auto">
<br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Messidor-2/test was not low, and dice_coef not high as shown below.
<br>
<pre>
loss,0.1583
dice_coef,0.8027
</pre>
<br>

<h3>
5 Tiled Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Messidor-2</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Messidor-2.<br>
<pre>
./4.tiledinfer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images (2240x1488 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(antillia ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled inferred test masks (2240x1488 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 2240x1488 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (antillia ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/images/20051020_45004_0100_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/masks/20051020_45004_0100_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test_output_tiled/20051020_45004_0100_PP.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/images/20051020_61998_0100_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/masks/20051020_61998_0100_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test_output_tiled/20051020_61998_0100_PP.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/images/20051116_44718_0400_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/masks/20051116_44718_0400_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test_output_tiled/20051116_44718_0400_PP.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/images/20051205_58373_0400_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/masks/20051205_58373_0400_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test_output_tiled/20051205_58373_0400_PP.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/images/20051212_36525_0400_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/masks/20051212_36525_0400_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test_output_tiled/20051212_36525_0400_PP.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/images/20051213_61892_0100_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test/masks/20051213_61892_0100_PP.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Messidor-2/mini_test_output_tiled/20051213_61892_0100_PP.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. Locating Blood Vessels in Retinal Images by Piecewise Threshold Probing of a Matched Filter Response</b><br>
Adam Hoover, Valentina Kouznetsova, and Michael Goldbaum<br>
<a href="https://www.uhu.es/retinopathy/General/000301IEEETransMedImag.pdf">
https://www.uhu.es/retinopathy/General/000301IEEETransMedImag.pdf
</a>
<br>
<br>
<b>2. High-Resolution Fundus (HRF) Image Database</b><br>
Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.<br>
<a href="https://www5.cs.fau.de/research/data/fundus-images/">
https://www5.cs.fau.de/research/data/fundus-images/
</a>.
<br>
<br>
<b>3. Robust Vessel Segmentation in Fundus Images</b><br>
Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.<br>

<a href="https://onlinelibrary.wiley.com/doi/10.1155/2013/154860">
https://onlinelibrary.wiley.com/doi/10.1155/2013/154860
</a>
<br>
<br>
<b>4. State-of-the-art retinal vessel segmentation with minimalistic models</b><br>
Adrian Galdran, André Anjos, José Dolz, Hadi Chakor, Hervé Lombaert & Ismail Ben Ayed<br>
<a href="https://www.nature.com/articles/s41598-022-09675-y">
https://www.nature.com/articles/s41598-022-09675-y
</a>
<br>
<br>
<b>5. Retinal blood vessel segmentation using a deep learning method based on modified U-NET model</b><br>
Sanjeewani, Arun Kumar Yadav, Mohd Akbar, Mohit Kumar, Divakar Yadav<br>
<a href="https://www.semanticscholar.org/reader/f5cb3b1c69a2a7e97d1935be9d706017af8cc1a3">
https://www.semanticscholar.org/reader/f5cb3b1c69a2a7e97d1935be9d706017af8cc1a3</a>
<br>
<br>
<b>6. Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</a>
<br>
<br>
<b>7, Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel</a>
<br>
<br>
