// FisioTech AI — Cámara como en este proyecto (funciona). Resto de lógica de carga/descarga desde KNNClassification.

let video;
const knnClassifier = ml5.KNNClassifier();
let poseNet;
let poses = [];

// Almacén para descargar archivos (como en carga)
let storeData = {
  dataA: [],
  dataB: [],
  result: []
};

function setup() {
  const canvas = createCanvas(640, 480);
  canvas.parent('videoContainer');
  video = createCapture(VIDEO);
  video.size(width, height);

  createButtons();

  poseNet = ml5.poseNet(video, modelReady);
  poseNet.on('pose', function(results) {
    poses = results;
  });
  video.hide();
}

function draw() {
  image(video, 0, 0, width, height);
  drawKeypoints();
  drawSkeleton();
}

function modelReady() {
  select('#status').html('Reconociendo Posiciones');
}

function download(filename, text) {
  const element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);
  element.style.display = 'none';
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
}

function downloadFilesHandler() {
  download('A.JSON', JSON.stringify(storeData.dataA));
  download('B.JSON', JSON.stringify(storeData.dataB));
  download('result.JSON', JSON.stringify(storeData.result));
}

function addExample(label) {
  const poseArray = poses[0].pose.keypoints.map(p => [p.score, p.position.x, p.position.y]);
  storeData['data' + label].push({ pose: poseArray });
  knnClassifier.addExample(poseArray, label);
  updateCounts();
}

function classify() {
  const numLabels = knnClassifier.getNumLabels();
  if (numLabels <= 0) {
    console.error('There is no examples in any label');
    return;
  }
  const poseArray = poses[0].pose.keypoints.map(p => [p.score, p.position.x, p.position.y]);
  knnClassifier.classify(poseArray, gotResults);
}

function createButtons() {
  select('#addClassA').mousePressed(function() { addExample('A'); });
  select('#addClassB').mousePressed(function() { addExample('B'); });
  select('#resetA').mousePressed(function() { clearLabel('A'); });
  select('#resetB').mousePressed(function() { clearLabel('B'); });
  select('#buttonPredict').mousePressed(classify);
  select('#clearAll').mousePressed(clearAllLabels);
  select('#downloadFiles').mousePressed(downloadFilesHandler);

  // Seleccionar archivo: botón abre el input file
  select('#btnFileA').mousePressed(function() { document.getElementById('fileInputA').click(); });
  select('#btnFileB').mousePressed(function() { document.getElementById('fileInputB').click(); });
  document.getElementById('fileInputA').addEventListener('change', function(e) { loadFileIntoLabel('A', e); });
  document.getElementById('fileInputB').addEventListener('change', function(e) { loadFileIntoLabel('B', e); });
}

function loadFileIntoLabel(label, e) {
  const file = e.target.files[0];
  const labelId = 'fileLabel' + label;
  if (!file) {
    select('#' + labelId).html('Sin archivo seleccionado');
    return;
  }
  const reader = new FileReader();
  reader.onload = function(ev) {
    try {
      const data = JSON.parse(ev.target.result);
      const arr = Array.isArray(data) ? data : (data.pose ? [data] : []);
      let added = 0;
      for (let i = 0; i < arr.length; i++) {
        const item = arr[i];
        const pose = item.pose || item;
        if (Array.isArray(pose) && pose.length > 0) {
          knnClassifier.addExample(pose, label);
          storeData['data' + label].push({ pose: pose });
          added++;
        }
      }
      updateCounts();
      select('#' + labelId).html(added ? file.name + ' (' + added + ' ejemplos)' : file.name + ' — formato no válido');
    } catch (err) {
      console.error(err);
      select('#' + labelId).html(file.name + ' — error al cargar');
    }
  };
  reader.readAsText(file);
  e.target.value = '';
}

function gotResults(err, result) {
  if (err) {
    console.error(err);
  }

  if (result.confidencesByLabel) {
    const confidences = result.confidencesByLabel;
    const resultStoreData = {
      result: '',
      confidence: 0,
      confidenceA: 0,
      confidenceB: 0
    };

    if (result.label) {
      select('#result').html(result.label);
      select('#confidence').html(confidences[result.label] * 100 + ' %');
      resultStoreData.result = result.label;
      resultStoreData.confidence = confidences[result.label] * 100;
    }

    select('#confidenceA').html((confidences['A'] ? confidences['A'] * 100 : 0) + ' %');
    select('#confidenceB').html((confidences['B'] ? confidences['B'] * 100 : 0) + ' %');
    resultStoreData.confidenceA = confidences['A'] ? confidences['A'] * 100 : 0;
    resultStoreData.confidenceB = confidences['B'] ? confidences['B'] * 100 : 0;
    storeData.result.push(resultStoreData);
  }

  classify();
}

function updateCounts() {
  const counts = knnClassifier.getCountByLabel();
  select('#exampleA').html(counts['A'] || 0);
  select('#exampleB').html(counts['B'] || 0);
}

function clearLabel(classLabel) {
  storeData['data' + classLabel] = [];
  knnClassifier.clearLabel(classLabel);
  updateCounts();
  select('#fileLabel' + classLabel).html('Sin archivo seleccionado');
}

function clearAllLabels() {
  storeData = {
    dataA: [],
    dataB: [],
    result: []
  };
  knnClassifier.clearAllLabels();
  updateCounts();
  select('#result').html('...');
  select('#confidence').html('...');
  select('#confidenceA').html('0');
  select('#confidenceB').html('0');
  select('#fileLabelA').html('Sin archivo seleccionado');
  select('#fileLabelB').html('Sin archivo seleccionado');
}

function drawKeypoints() {
  for (let i = 0; i < poses.length; i++) {
    const pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      const keypoint = pose.keypoints[j];
      if (keypoint.score > 0.2) {
        fill(255, 0, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

function drawSkeleton() {
  for (let i = 0; i < poses.length; i++) {
    const skeleton = poses[i].skeleton;
    for (let j = 0; j < skeleton.length; j++) {
      const partA = skeleton[j][0];
      const partB = skeleton[j][1];
      stroke(255, 0, 0);
      strokeWeight(2);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}
