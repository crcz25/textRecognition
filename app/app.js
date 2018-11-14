let model;

document.getElementById('chart_box').innerHTML = "";
document.getElementById('chart_box').style.display = "none";

async function loadModel() {
  console.log("model loading..");
  model = await tf.loadModel('https://raw.githubusercontent.com/crcz25/textRecognition/develop/output/model.json');
  // console.log(model)
  console.log("model loaded..");
}

loadModel();

function preprocessCanvas(image) {
	console.log("PREPROCESS")

    let tensor = tf.fromPixels(image).mean(2).reshape([1, 784]).toFloat()
	console.log(tensor)
    return tensor.div(255.0);
}

function cloneCanvas(oldCanvas) {

    //create a new canvas
    var newCanvas = document.createElement('canvas');
	var context = newCanvas.getContext('2d');


    //set dimensions
    newCanvas.width = oldCanvas.width / 8;
	newCanvas.height = oldCanvas.height / 8;

    //apply the old canvas to the new one
    context.drawImage(oldCanvas, 0, 0, oldCanvas.width/8, oldCanvas.height/8);

    //return the new canvas
    return newCanvas;
}


async function predict() {

	// get image data from canvas
	var imageData = canvas.toDataURL();

	var cln = cloneCanvas(canvas)
	console.log(cln)
	console.log(cln.toDataURL())

	// preprocess canvas
    let tensor = preprocessCanvas(cln);
    console.log(tensor)

	// make predictions on the preprocessed image tensor
	let predictions = await model.predict(tensor).data();

	// get the model's prediction results
	let results = Array.from(predictions)

	// display the predictions in chart
	displayChart(results)

	console.log(results);
}

//------------------------------
// Chart to display predictions
//------------------------------
var chart = "";
var firstTime = 0;
function loadChart(label, data, modelSelected) {
	var ctx = document.getElementById('chart_box').getContext('2d');
	chart = new Chart(ctx, {
	    // The type of chart we want to create
	    type: 'bar',

	    // The data for our dataset
	    data: {
	        labels: label,
	        datasets: [{
	            label: modelSelected + " prediction",
	            backgroundColor: '#f50057',
	            borderColor: 'rgb(255, 99, 132)',
	            data: data,
	        }]
	    },

	    // Configuration options go here
	    options: {}
	});
}

function displayChart(data) {

	label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
	if (firstTime == 0) {
		loadChart(label, data, "CNN");
		firstTime = 1;
	} else {
		chart.destroy();
		loadChart(label, data, "CNN");
	}
	document.getElementById('chart_box').style.display = "block";
}