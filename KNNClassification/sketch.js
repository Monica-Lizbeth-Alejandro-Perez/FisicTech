// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
KNN Classification on Webcam Images with poseNet. Built with p5.js
=== */
let video;
// Create a KNN classifier
const knnClassifier = ml5.KNNClassifier();
let poseNet;
let poses = [];
var strokeWeightElement = document.getElementById("strokeWeight");
var strokeLineColor = document.getElementById("strokeLineColor");
var storeData = {
  dataA: [],
  dataB: [],
  result: [],
};

function setup() {
  const canvas = createCanvas(576, 324);
  canvas.parent("videoContainer");
  video = createCapture(VIDEO);
  video.size(width, height);

  // Create the UI buttons
  createButtons();

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, modelReady);
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on("pose", function (results) {
    poses = results;
  });
  // Hide the video element, and just show the canvas
  video.hide();
}

function draw() {
  image(video, 0, 0, width, height);

  // We can call both functions to draw all keypoints and the skeletons
  drawKeypoints();
  drawSkeleton();
}

function modelReady() {
  select("#status").html("Reconociendo Posiciones");
}
function download(filename, text) {
  var element = document.createElement("a");
  element.setAttribute(
    "href",
    "data:text/plain;charset=utf-8," + encodeURIComponent(text)
  );
  element.setAttribute("download", filename);

  element.style.display = "none";
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

// Add the current frame from the video to the classifier
function addExample(label) {
  // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
  const poseArray = poses[0].pose.keypoints.map((p) => [
    p.score,
    p.position.x,
    p.position.y,
  ]);
  storeData[`data${label}`].push({ pose: poseArray });

  // Add an example with a label to the classifier
  // Start file download.
  //download(`${label}.JSON`, JSON.stringify(poseArray));

  knnClassifier.addExample(poseArray, label);
  updateCounts();
}

// Predict the current frame.
function classify() {
  // Get the total number of labels from knnClassifier
  const numLabels = knnClassifier.getNumLabels();
  if (numLabels <= 0) {
    console.error("There is no examples in any label");
    return;
  }
  // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
  const poseArray = poses[0].pose.keypoints.map((p) => [
    p.score,
    p.position.x,
    p.position.y,
  ]);

  // Use knnClassifier to classify which label do these features belong to
  // You can pass in a callback function `gotResults` to knnClassifier.classify function
  knnClassifier.classify(poseArray, gotResults);
}

// A util function to create UI buttons
function createButtons() {
  // When the A button is pressed, add the current frame
  // from the video with a label of "A" to the classifier
  buttonA = select("#addClassA");
  buttonA.mousePressed(function () {
    addExample("A");
  });

  // When the B button is pressed, add the current frame
  // from the video with a label of "B" to the classifier
  buttonB = select("#addClassB");
  buttonB.mousePressed(function () {
    addExample("B");
  });

  // Reset buttons
  resetBtnA = select("#resetA");
  resetBtnA.mousePressed(function () {
    clearLabel("A");
  });

  resetBtnB = select("#resetB");
  resetBtnB.mousePressed(function () {
    clearLabel("B");
  });

  // Predict button
  buttonPredict = select("#buttonPredict");
  buttonPredict.mousePressed(classify);

  // Clear all classes button
  buttonClearAll = select("#clearAll");
  buttonClearAll.mousePressed(clearAllLabels);

  // downloadAndClear

  downloadAndClear = select("#downloadAndClear");
  downloadAndClear.mousePressed(downloadAndClearHandler);
}
function downloadAndClearHandler() {
  download("A.JSON", JSON.stringify(storeData.dataA));
  download("B.JSON", JSON.stringify(storeData.dataB));
  download("result.JSON", JSON.stringify(storeData.result));
}
// Show the results
function gotResults(err, result) {
  // Display any error
  if (err) {
    console.error(err);
  }

  if (result.confidencesByLabel) {
    let resultStoreData = {
      result: "",
      confidence: 0,
      confidenceA: 0,
      confidenceB: 0,
    };
    const confidences = result.confidencesByLabel;
    // result.label is the label that has the highest confidence
    if (result.label) {
      select("#result").html(result.label);
      select("#confidence").html(`${confidences[result.label] * 100} %`);
      resultStoreData.result = result.label;
      resultStoreData.confidence = confidences[result.label] * 100;
    }

    select("#confidenceA").html(
      `${confidences["A"] ? confidences["A"] * 100 : 0} %`
    );

    select("#confidenceB").html(
      `${confidences["B"] ? confidences["B"] * 100 : 0} %`
    );
    resultStoreData.confidenceA = confidences["A"] ? confidences["A"] * 100 : 0;
    resultStoreData.confidenceB = confidences["B"] ? confidences["B"] * 100 : 0;
    storeData.result.push(resultStoreData);
  }
  classify();
}

// FUNCIONES DE CARGA EXTERNA
function mostrarContenido(contenido) {
  const elemento = document.getElementById('contenido-archivo');
  elemento.innerHTML = contenido;
} 
-->

function leerArchivo(e) {
const archivo = e.target.files[0];
if (!archivo) {
  return;
}
const lector = new FileReader();
lector.onload = function(e) {
  const contenido = e.target.result;
  mostrarContenido(contenido);
};
lector.readAsText(archivo);
}


document.querySelector('#archivo1')
.addEventListener('change', leerArchivo, false);


// TERMINAN FUNCIONES DE CARGA EXTERNA


// Update the example count for each label
function updateCounts() {
  const counts = knnClassifier.getCountByLabel();

  select("#exampleA").html(counts["A"] || 0);
  select("#exampleB").html(counts["B"] || 0);
}

// Clear the examples in one label
function clearLabel(classLabel) {
  storeData[`data${classLabel}`] = [];
  knnClassifier.clearLabel(classLabel);
  updateCounts();
}

// Clear all the examples in all labels
function clearAllLabels() {
  storeData = {
    dataA: [],
    dataB: [],
    result: [],
  };
  knnClassifier.clearAllLabels();
  updateCounts();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      let keypoint = pose.keypoints[j];
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill(255, 0, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

const hexToRgb = (hex) => {
  try {
    return hex
      .replace(
        /^#?([a-f\d])([a-f\d])([a-f\d])$/i,
        (m, r, g, b) => "#" + r + r + g + g + b + b
      )
      .substring(1)
      .match(/.{2}/g)
      .map((x) => parseInt(x, 16));
  } catch (error) {
    return [0, 51, 255];
  }
};

// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected

  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(...hexToRgb(strokeLineColor?.value));
      strokeWeight(+strokeWeightElement?.value || 4);
      line(
        partA.position.x,
        partA.position.y,
        partB.position.x,
        partB.position.y
      );
    }
  }
}
