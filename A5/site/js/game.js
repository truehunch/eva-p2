const videoWidth = 600;
const videoHeight = 500;

var isMobile = () => true;
/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const defaultQuantBytes = 2;

const defaultArchitecture = 'MobileNetV1'
const defaultMobileNetMultiplier = isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
// [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]);
const defaultMobileNetInputResolution = 500;

const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 250;

const varDefaultLoc = null;

var left = ["leftEye", "leftEar", "leftShoulder", "leftElbow", "leftWrist", "leftHip", "leftKnee", "leftAnkle"];
var right = ["rightEye", "rightEar", "rightShoulder", "rightElbow", "rightWrist", "rightHip", "rightKnee", "rightAnkle"];

const partLength = 8;

var defalutKeyPoints = {
  "nose": varDefaultLoc,
  "leftEye": varDefaultLoc,
  "rightEye": varDefaultLoc,
  "leftEar": varDefaultLoc,
  "rightEar": varDefaultLoc,
  "leftShoulder": varDefaultLoc,
  "rightShoulder": varDefaultLoc,
  "leftElbow": varDefaultLoc,
  "rightElbow": varDefaultLoc,
  "leftWrist": varDefaultLoc,
  "rightWrist": varDefaultLoc,
  "leftHip": varDefaultLoc,
  "rightHip": varDefaultLoc,
  "leftKnee": varDefaultLoc,
  "rightKnee": varDefaultLoc,
  "leftAnkle": varDefaultLoc,
  "rightAnkle": varDefaultLoc
};

var currentKeyPoints = defalutKeyPoints;
var previousKeyPoints = null;
var slopes = [];
var slopeWindow = 0;
const windowSize = 2;

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');

  // since images are being fed from a webcam, we want to feed in the
  // original image and then just flip the keypoints' x coordinates. If instead
  // we flip the image, then correcting left-right keypoint pairs requires a
  // permutation on all the keypoints.
  const flipPoseHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;


  async function poseDetectionFrame() {
      ctx.clearRect(0, 0, videoWidth, videoHeight);

      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();

      const pose = await net.estimatePoses(video, {
        flipHorizontal: flipPoseHorizontal,
        decodingMethod: 'single-person'
      });

      keypoints = pose[0]['keypoints']

      // console.log(pose)
      keypoints.forEach((keypoint) => {
        if(keypoint['score'] > 0.75){
          ctx.beginPath();
          ctx.arc(keypoint['position']['x'], keypoint['position']['y'], 5, 0, 2 * Math.PI);
          ctx.lineWidth = 2;
          ctx.strokeStyle = '#FFFFFF';
          ctx.stroke();
        }
      });

      previousKeyPoints = keypoints;
      // console.log(keypoints);

      keypoints.forEach((keypoint) => {
        if(keypoint['score'] > 0.75){
          ctx.beginPath();
          ctx.arc(keypoint['position']['x'], keypoint['position']['y'], 5, 0, 2 * Math.PI);
          ctx.lineWidth = 2;
          ctx.strokeStyle = '#FFFFFF';
          ctx.stroke();
          currentKeyPoints[keypoint['part']] = keypoint['position'];
        } else {
          currentKeyPoints[keypoint['part']] = null;
        }
      });

      if(previousKeyPoints != null){
        var slope = 0.0, valid_count = 0;
        for(i = 0; i < partLength; i++){
          letfPart = left[i];
          rightPart = right[i];
          if(currentKeyPoints[letfPart] != null & currentKeyPoints[rightPart] != null){
            var left_x = currentKeyPoints[letfPart]['x'];
            var left_y = currentKeyPoints[letfPart]['y'];
            var right_x = currentKeyPoints[rightPart]['x'];
            var right_y = currentKeyPoints[rightPart]['y'];
            slope_part = (left_y - right_y) / (left_x - right_x);
            if(!Number.isNaN(slope_part)){
              slope += slope_part;
              valid_count += 1;
            }
          }
        }
        if (valid_count > 0) {
          slope = slope / valid_count;
          if(Math.abs(slope) > 0.1){
            slopes.push(slope > 0 ? 1 : -1);
          } else {
            slopes.shift();
          }
          var control = slopes.reduce((a, b) => a + b, 0);
          console.log(control);

          if (control > 0){
            rightPressed = true;
            leftPressed = false;
          } else if (control < 0) {
            rightPressed = false;
            leftPressed = true;
          } else {
            rightPressed = false;
            leftPressed = false;
          }
        }

        if(slopes.length > windowSize){
          slopes = [];
        }


      }

      previousKeyPoints = currentKeyPoints;
      
      adjacent_points = posenet.getAdjacentKeyPoints(keypoints, 0.75);
      adjacent_points.forEach((points) => {
        p1 = points[0]['position']
        p2 = points[1]['position']
        ctx.beginPath();
        ctx.moveTo(p1['x'], p1['y']);
        ctx.lineTo(p2['x'], p2['y']);
        ctx.lineWidth = 1;
        ctx.strokeStyle = '#FF0000';
        ctx.stroke();
      });

    setTimeout(()=>requestAnimationFrame(poseDetectionFrame), 1000 / 30)
  }

  poseDetectionFrame();

}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
async function bindPage() {

  const net = await posenet.load({
      architecture: defaultArchitecture,
      outputStride: defaultMobileNetStride,
      inputResolution: defaultMobileNetInputResolution,
      multiplier: defaultMobileNetMultiplier,
    });

  console.log('Posenet loaded...')
  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }
  detectPoseInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();

var canvas = document.getElementById("game");
var ctx = canvas.getContext("2d");
var ballRadius = 10;
var x = canvas.width/2;
var y = canvas.height-30;
var dx = 1;
var dy = -1;
var paddleHeight = 10;
var paddleWidth = 100;
var paddleX = (canvas.width-paddleWidth)/2;
var rightPressed = false;
var leftPressed = false;
var brickRowCount = 5;
var brickColumnCount = 3;
var brickWidth = 75;
var brickHeight = 20;
var brickPadding = 10;
var brickOffsetTop = 30;
var brickOffsetLeft = 30;
var score = 0;
var lives = 4;

function reset(){
  score = 0;
  lives = 4;
  // paddleX = (canvas.width-paddleWidth)/2;

  for(var c=0; c<brickColumnCount; c++) {
    for(var r=0; r<brickRowCount; r++) {
      bricks[c][r]['status'] = 1;
    }
  }
}

$("#play-btn").on("click", () => 
{
  console.log('Playing');
  $("#play-btn").prop('disabled', true);
  draw();
});

function init_ui(){
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawBricks();
  drawBall();
  drawPaddle();
  drawScore();
  drawLives();
}

var bricks = [];
for(var c=0; c<brickColumnCount; c++) {
  bricks[c] = [];
  for(var r=0; r<brickRowCount; r++) {
    bricks[c][r] = { x: 0, y: 0, status: 1 };
  }
}

document.addEventListener("keydown", keyDownHandler, false);
document.addEventListener("keyup", keyUpHandler, false);
//document.addEventListener("mousemove", mouseMoveHandler, false);

function keyDownHandler(e) {
    if(e.key == "Right" || e.key == "ArrowRight") {
        rightPressed = true;
    }
    else if(e.key == "Left" || e.key == "ArrowLeft") {
        leftPressed = true;
    }
}

function keyUpHandler(e) {
    if(e.key == "Right" || e.key == "ArrowRight") {
        rightPressed = false;
    }
    else if(e.key == "Left" || e.key == "ArrowLeft") {
        leftPressed = false;
    }
}

function mouseMoveHandler(e) {
  var relativeX = e.clientX - canvas.offsetLeft;
  if(relativeX > 0 && relativeX < canvas.width) {
    paddleX = relativeX - paddleWidth/2;
  }
}
function collisionDetection() {
  for(var c=0; c<brickColumnCount; c++) {
    for(var r=0; r<brickRowCount; r++) {
      var b = bricks[c][r];
      if(b.status == 1) {
        if(x > b.x && x < b.x+brickWidth && y > b.y && y < b.y+brickHeight) {
          dy = -dy;
          b.status = 0;
          score++;
          if(score == brickRowCount*brickColumnCount) {
            $("#game-popup-status").html("You win!!");
            $("#game-popup-img").attr("src", "assets/img/crown.svg");
            $('#game-popup').modal();
            reset();
            return true;
            // alert("YOU WIN, CONGRATS!");
            // document.location.reload();
          }
        }
      }
    }
  }
  return false;
}

function drawBall() {
  ctx.beginPath();
  ctx.arc(x, y, ballRadius, 0, Math.PI*2);
  ctx.fillStyle = "#0095DD";
  ctx.fill();
  ctx.closePath();
}
function drawPaddle() {
  ctx.beginPath();
  ctx.rect(paddleX, canvas.height-paddleHeight, paddleWidth, paddleHeight);
  ctx.fillStyle = "#0095DD";
  ctx.fill();
  ctx.closePath();
}

function drawBricks() {
  for(var c=0; c<brickColumnCount; c++) {
    for(var r=0; r<brickRowCount; r++) {
      if(bricks[c][r].status == 1) {
        var brickX = (r*(brickWidth+brickPadding))+brickOffsetLeft;
        var brickY = (c*(brickHeight+brickPadding))+brickOffsetTop;
        bricks[c][r].x = brickX;
        bricks[c][r].y = brickY;
        ctx.beginPath();
        ctx.rect(brickX, brickY, brickWidth, brickHeight);
        ctx.fillStyle = "#0095DD";
        ctx.fill();
        ctx.closePath();
      }
    }
  }
}

function drawScore() {
  ctx.font = "16px Arial";
  ctx.fillStyle = "#0095DD";
  ctx.fillText("Score: "+score, 8, 20);
}

function drawLives() {
  ctx.font = "16px Arial";
  ctx.fillStyle = "#0095DD";
  ctx.fillText("Lives: "+lives, canvas.width-65, 20);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  drawBricks();
  drawBall();
  drawPaddle();
  drawScore();
  drawLives();
  if(collisionDetection()){
    $("#play-btn").prop('disabled', false);
    return;
  }

  if(x + dx > canvas.width-ballRadius || x + dx < ballRadius) {
    dx = -dx;
  }
  if(y + dy < ballRadius) {
    dy = -dy;
  }
  else if(y + dy > canvas.height-ballRadius) {
    if(x > paddleX && x < paddleX + paddleWidth) {
      dy = -dy;
    }
    else {
      lives--;
      if(!lives) {
        $("#game-popup-status").html("Game over!!");
        $("#game-popup-img").attr("src", "assets/img/fail.svg");
        $('#game-popup').modal();
        reset();
        $("#play-btn").prop('disabled', false);
        return;
        // alert("GAME OVER");
        // document.location.reload();
      }
      else {
        x = canvas.width/2;
        y = canvas.height-30;
        dx = 2;
        dy = -2;
        // paddleX = (canvas.width-paddleWidth)/2;
      }
    }
  }

  if(rightPressed && paddleX < canvas.width-paddleWidth) {
    paddleX += 4;
  }
  else if(leftPressed && paddleX > 0) {
    paddleX -= 4;
  }

  x += dx;
  y += dy;
  requestAnimationFrame(draw);
}

// draw();
init_ui();