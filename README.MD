Introduction: 

Information theory is a big topic and I've been thinking at how it intersects with image classification. Looking at modern image recognition, we use CNNs via deep learning. CNNs work by sliding a 2d box over an image to down sample and extract useful info from an image. I have a few questions id like to state: 

How much of the image is truly needed? 
Is there a way to reveal how sparse data truly is? 
Do we care more about having an intelligent system or better performance? 

My first two questions dance around the overall theme for this topic in which how can we minimize the amount of data needed to accurately understand something. Think about telling someone to guess whether a picture is a cat or a dog based on a few descriptions you give them. You wouldn't say this animal has fur or sharp teeth but rather they have whiskers or they have a long snout. My last point is that maybe we give machines too much discretion.

Methodology: 

My overall question is general enough to be applied in many different ways but for this experiment, we can use image classification for its simplicity and feasibility. 

Base Experiment: 

For a bare bones experiment, I present this: 

Take an image dataset, perform edge detection on each image turning it into a sparse matrix, record metadata such as color/gradients, perform image recognition. 

Images are very dense in information but the density is not necessarily indicative of a true signal for our task, recognition. Humans operate exceptionally well with dense information but computers are not human, over half of the work we do when building CNNs is attempting to make them see the world how a human does which naturally comes with lots of data processing and algorithmic efficiency being less than ideal. 

Edge Detection: 

Methodology:
This will turn our image into a near binary image, blacking out the background and solely focusing on the edges that truly matter for our task. This massively reduces the size of the data but also how fast a computer can analyze it. We can utilize sparse matrices and the programs that make analyzing them faster to take advantage of our pipeline. 

I chose 2 algorithms: Sobel and Canny.

Running these alone will allow for the background to capture unneeded attention from downstream algorithms which is why I want to isolate the object itself via a mask. I made a function to draw the image in a box and this can be automated via some object/anomaly detection model,

Results:
In terms of raw performance, I'd say Sobel performed the best in capturing edges but Canny focused on the target more which is what I do want. When adding the mask, Sobel leaves more information than Canny and I'm unsure of which is better. My object for reference is a bunny so sobel leaves the fur more or less filled in. As for classification from other animals, the fur may be unneeded but if you want to differentiate between different species of bunnies, the fur texture and pattern may be useful.

File size Images (PNG Format, Size on Disk): 

Full Image: 200 KB
Masked Image: 80 KB 
Sobel w/o Mask: 92 KB 
Sobel w/ Mask: 40 KB 
Canny w/o Mask: 8 KB 
Canny w/ Mask: 4 KB

This is the limit of size optimization that we can do as saving them as sparse matrices only contributes to algorithmic performance for downstream algorithms. Meaning converting canny.png to canny.npz doesn't offer any less file size.

Metadata: 

This metadata contains things such as color, this would either be encoded into the sparse matrix itself such as defining the color of the edges or as a separate structure which can go into more depth. I suggest this as converting a dense matrix into a sparse one will inevitably lose information and lots of it so this can ideally conserve the majority of information loss without it being a computational bottleneck. 

Methodology: 
I have a few ways to do this such as simply using color instead of white to line the edges of an image or running a kernel over the cropped image to preserve general regions of color. In a normal workflow, the edge detection and color detection would be run in parallel with one pass over the data to save time. 

Image Recognition: 

Like CNNs, the final step is to simply run this through an ML algorithm such as a Neural Network but with more simplified data, other algorithms may be of use such as tree based ones given the nature of the data. I want to utilize the nature of sparse matricies to avoid losing computation to the black regions of the masked images.