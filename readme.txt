The file faceDet is the first impelementation of the face detection and the hand tracking. Here I wanted to use my own model for robust face recognition, as I wanted to Check if ratio of face width and height changes much from original, then probably CAMShift needs updates and also to avoid the error from                  background changes, where old face histogram is no longer a good representation of the face. However due to some error I could not finish the further modification on that. So I try to final submssion improve my code.

For now I use the FaceFinal.py code.
It takes the original codes from opencv library as the face_detect adn camshift. Also uses the functions as calcHist()

The code is well commented. But the general workflow is like, 
1) video camera frame taken-> grayscale
2) feed into face_detect, rects[] -> contain 4 points of rectangle containing the face that is found. Drawing rectangle on the found area.
3) Calculating the histogram, (but firstly converting into hsv) 
4) When hist of face color is calculated, calculate prob of each pixel belongong to that prob or not. 
5) Face region pixels are set to 0, to leave only the pixels belonging to other things that mostly resembles skin color
6) Now calling camshift to track that skin color, it will take into tracking the area with the biggest prob of having that skin color.
7) Finding that area, taking the roi coordinates, drawing ellipse. 

Now we can press "s" (only after the hand was detected, otherwise will return error)
Then we get the hand box, we can store it as resized 16 by 16 image. Then we feed it into Multilayer Peceptron (nn). 
But I had restricted time so took the already made dataset, one is letter.txt another is letter_use.txt. In that files images were flattened, so pixel values for 16 byb16 image were converted into 1D and all is put into txt file.
Then with the help of the ready mlp impelemnattion, I trained data. Got accuracy more that 99. And have 2 models one is model.txt and one is model_lettertrained on letter.txt). 
Then in the main function after grabbing the roi of the hand I can call model.predict, but before need to load the ready model into my file.
Then we can predict the letters!

