const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let net;

async function app() {
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');

    // Make a prediction of image from webcam
    const webcam = await tf.data.webcam(webcamElement);

    const addExampleToKNN = async classId => {
        // capture an image from the web camera
        const img = await webcam.capture();

        // get the intermediate activation of MobileNet 'conv_preds' and pass that to KNN classifier instance
        const activation = net.infer(img, true);

        // pass the intermediate activation to the classifier
        classifier.addExample(activation, classId);

        // dispose the tensor to release memory
        img.dispose()
    }

    document.getElementById('class-a').addEventListener('click', () => addExampleToKNN(0));
    document.getElementById('class-b').addEventListener('click', () => addExampleToKNN(1));
    document.getElementById('class-c').addEventListener('click', () => addExampleToKNN(2));

    while (true) {
        if(classifier.getNumClasses() > 0) {
            const img = await webcam.capture();

            // get activation of image from webcam
            const activation = net.infer(img, 'conv_preds')
            console.log('activation:  ', activation)

            const result = await classifier.predictClass(activation);

            const classes = ['A', 'B', 'C']

            document.getElementById('console').innerText = `
                prediction: ${classes[result.label]}\n
                probability: ${result.confidences[result.label]}
        `
            // dispose the tensor to release the memory
            img.dispose();

        }
        await tf.nextFrame();
    }
}

app();