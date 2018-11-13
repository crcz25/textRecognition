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
    let tensor = tf.fromPixels(image)
        .resizeNearestNeighbor([28, 28])
        .mean(2)
        .expandDims(2)
        .expandDims()
        //.reshape([28, 28, 1])
        .toFloat();
    console.log(tensor.shape);
    return tensor.div(255.0);
}

async function predict() {

	// get image data from canvas
	var imageData = canvas.toDataURL();
	console.log(imageData)

	// preprocess canvas
    let tensor = preprocessCanvas(canvas);
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