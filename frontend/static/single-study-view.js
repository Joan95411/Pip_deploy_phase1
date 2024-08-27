
async function backToAllStudies() {
    window.location.href = "/study/all";
}
function nextStudy() {
    // Get the current study's accession number
    const currentAccessionNumber = document.getElementById('current-accession-number').value;
    console.log(currentAccessionNumber);
    // Create the URL for the next study
    const nextStudyUrl = `/study/${currentAccessionNumber}/next`;

    // Redirect to the next study
    window.location.href = nextStudyUrl;
}
function previousStudy() {
    // Get the current study's accession number
    const currentAccessionNumber = document.getElementById('current-accession-number').value;
    console.log(currentAccessionNumber);
    // Create the URL for the next study
    const nextStudyUrl = `/study/${currentAccessionNumber}/previous`;

    // Redirect to the next study
    window.location.href = nextStudyUrl;
}
function customAlert(msg,duration)
{
    const styler = document.createElement("div");
    styler.setAttribute("style","border: solid 5px Red;width:auto;height:auto;top:50%;left:40%;background-color:#444;color:Silver");
 styler.innerHTML = "<h1>"+msg+"</h1>";
 setTimeout(function()
 {
   styler.parentNode.removeChild(styler);
 },duration);
 document.body.appendChild(styler);
}


async function displayPrototypesAndImages(prototypes_json, fracturedImagePath) {
    await displayImage( fracturedImagePath);
    await displayPrototypes(prototypes_json,fracturedImagePath);
    const normalizedPath = fracturedImagePath.replace(/\\/g, '/');
    console.log('Normalized fracturedImagePath:', normalizedPath);
}

function flagErrorToDatabase(prototype_uid) {
    // Implement the logic to flag the error in the database
    const endPoint = `/study/flag_error/${prototype_uid}`
    fetch(endPoint, {
        method: "PATCH",
    })
    .then(response => {
        if (!response.ok) {
            alert("Error flagging error in the database. Please try again")
        } else {
            alert("Error flagged successfully. It might appear on this page as false until the next refresh, " +
                    "but if you see this message, it has been flagged in the database.")
}
    })
    .then(result => {
        // Handle the response from the server if needed
        console.log("Server response:", result);
    })
    .catch(error => {
        console.error("Error:", error);
    });
}

function removeCanvas() {
    if (canvas) {
        canvas.remove();  // Remove the canvas from the DOM
        canvas = null;    // Clear the canvas variable
        context = null;   // Clear the context variable
        clicks = [];      // Clear the clicks array
        const addPrototypeButton = document.querySelector('.add-prototype-button');

    // Check if we're in the process of drawing
    if (addPrototypeButton.textContent === 'Finish Drawing') {
        addPrototypeButton.textContent = 'Add Prototype';
        return;
    }
    }

}
async function displayPrototypes(prototypes_json,fracturedImagePath) {

    const prototypes = JSON.parse(prototypes_json); // Parse the JSON string into an array
    const prototypesBar = document.getElementById('prototypes-bar');
    prototypesBar.innerHTML = ''; // Clear previous content
    const prototypeTable = document.createElement('table');
    prototypeTable.classList.add('prototype-table');

    const prototypeHeader = document.createElement('h2');
    prototypeHeader.classList.add('prototype-header');
    prototypeHeader.textContent = 'Prototypes in this image';

    prototypesBar.appendChild(prototypeHeader);

    const headerRow = prototypeTable.insertRow();
    const headerCellID = headerRow.insertCell();
    headerCellID.textContent = 'Index';
    const headerCellSimWeight = headerRow.insertCell();
    headerCellSimWeight.textContent = 'Similarity Weight';
    const headerCellPredClass = headerRow.insertCell();
    headerCellPredClass.textContent = 'Predicted Class';
    const headerCellFlaggedError = headerRow.insertCell();
    headerCellFlaggedError.textContent = 'Flagged Error';
    const headerCellManuallyAnnotated = headerRow.insertCell();
    headerCellManuallyAnnotated.textContent = 'Manually Annotated';

    prototypes.forEach(function(prototype) {

        const dataRow = prototypeTable.insertRow();
        const dataCellID = dataRow.insertCell();
        // dataCellID.textContent = prototype.prototype_index;
        const indexButton = document.createElement('button');
        indexButton.textContent = prototype.prototype_index;
        const predictClass=prototype.predicted_class.toLowerCase();
        // Add a click event listener to the index button
        indexButton.addEventListener('click', function () {
            displayPrototypeImage(prototype.prototype_index,fracturedImagePath,predictClass); // Call the function to display the image
        });
        dataCellID.appendChild(indexButton);

        const dataCellSimWeight = dataRow.insertCell();
        dataCellSimWeight.textContent = prototype.similarity_weight;
        const dataCellPredClass = dataRow.insertCell();
        dataCellPredClass.textContent = prototype.predicted_class;

        const dataCellFlaggedError = dataRow.insertCell();
        const isFlagged = prototype.flagged_error === 1;
        const buttonElement = document.createElement('button');
        buttonElement.textContent = isFlagged ? 'True' : 'False';

        // Add a click event listener to the button
        buttonElement.addEventListener('click', function () {
            if (buttonElement.textContent === 'False') {
                buttonElement.textContent = 'True';
                flagErrorToDatabase(prototype.prototype_uid);
            }
        });

    // Append the button to the cell
    dataCellFlaggedError.appendChild(buttonElement);

    const dataCellManuallyAnnotated = dataRow.insertCell();
    if (prototype.manually_annotated === 1) {
        dataCellManuallyAnnotated.textContent = "True";
    } else {
        dataCellManuallyAnnotated.textContent = "False";
    }

    prototypesBar.appendChild(prototypeTable);
});
// Create the "Add Prototype" button
    const addPrototypeButton = document.createElement('button');
    addPrototypeButton.textContent = 'Add Prototype';
    addPrototypeButton.classList.add('add-prototype-button');

    // Add event listener for the button
    addPrototypeButton.addEventListener('click', function () {
        addNewPrototype();
    });

    // Append the button below the table
    prototypesBar.appendChild(addPrototypeButton);
}
var clicks = [];
var canvas, context, imageObj;

function addNewPrototype() {
    const addPrototypeButton = document.querySelector('.add-prototype-button');

    // Check if we're in the process of drawing
    if (addPrototypeButton.textContent === 'Finish Drawing') {
        // Save the drawing and reset the button
        saveDrawing();
        addPrototypeButton.textContent = 'Add Prototype';
        return;
    }

    // Change button text to 'Finish Drawing'
    addPrototypeButton.textContent = 'Finish Drawing';
    var imageDisplay = document.querySelector('.image-display');

    // Remove any existing canvas if present
    var existingCanvas = document.getElementById('drawing-canvas');
    if (existingCanvas) {
        imageDisplay.removeChild(existingCanvas);
    }

    // Create a new canvas element
    canvas = document.createElement('canvas');
    canvas.id = 'drawing-canvas';
    imageDisplay.appendChild(canvas);

    context = canvas.getContext('2d');
    imageObj = new Image();

    // Load the image and set canvas dimensions
    imageObj.onload = function() {
        $(canvas).attr({
            width: this.width,
            height: this.height
        });
        context.drawImage(imageObj, 0, 0);
    };


    // Set the image source to the x-ray image
    var imgElement = document.getElementById('x-ray-image');
    imageObj.src = imgElement.src;
    // Reset clicks array if the button is clicked again
    clicks = [];

    // Set up the event listener for drawing points on the canvas
    canvas.addEventListener('mouseup', function(e){
        const rect = canvas.getBoundingClientRect();

        // Calculate the cursor position relative to the canvas
        // Since the canvas is scaled, calculate the scale factors
        const scaleX = imageObj.width / rect.width;
        const scaleY = imageObj.height / rect.height;

        // Correctly map the mouse position to the canvas coordinates
        const offsetX = (e.clientX - rect.left) * scaleX;
        const offsetY = (e.clientY - rect.top) * scaleY;

        // Print the cursor offset
        console.log('Cursor X: ' + offsetX + ', Cursor Y: ' + offsetY);

        clicks.push({
            x: offsetX,
            y: offsetY
        });
        redraw();
    });
}

function saveDrawing() {
    // Convert the canvas content (image + drawing) to a data URL
    const dataURL = canvas.toDataURL('image/png');

    // Create an anchor element to download the image
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = 'annotated_image.png';

    // Trigger the download
    link.click();

    console.log('Drawing saved as a new image.');
    removeCanvas();
}
function drawPolygon() {
    context.fillStyle = 'rgba(100,100,100,0.5)';
    context.strokeStyle = "#df4b26";
    context.lineWidth = 1;

    context.beginPath();
    context.moveTo(clicks[0].x, clicks[0].y);
    for (var i = 1; i < clicks.length; i++) {
        context.lineTo(clicks[i].x, clicks[i].y);
    }
    context.closePath();
    context.fill();
    context.stroke();
}

// Function to draw points on the canvas
function drawPoints() {
    context.strokeStyle = "#df4b26";
    context.lineJoin = "round";
    context.lineWidth = 10;

    for (var i = 0; i < clicks.length; i++) {
        context.beginPath();
        context.arc(clicks[i].x, clicks[i].y, 3, 0, 2 * Math.PI, false);
        context.fillStyle = '#ffffff';
        context.fill();
        context.lineWidth = 5;
        context.stroke();
    }
}

// Function to redraw the canvas
function redraw() {
    // Clear the canvas
    canvas.width = canvas.width;
    // Redraw the image
    context.drawImage(imageObj, 0, 0);
    // Draw the polygon and points
    drawPolygon();
    drawPoints();
}

// Function to display the prototype image
async function displayPrototypeImage(prototypeIndex,fracturedImagePath,predictClass) {
    removeCanvas();
    let fracturedImagePath3;
    try {
        const clickedButton = event.currentTarget;
        // Remove the 'active' class from all buttons
        const allButtons = document.querySelectorAll('.right-sidebar button');
        allButtons.forEach(function (button) {
            button.classList.remove('active');
        });
        const commentSection = document.querySelector('.comment-section');
        // Add the 'active' class to the clicked button
        clickedButton.classList.add('active');

        const lastSlashIndex = fracturedImagePath.lastIndexOf('\\'); // Find the index of the last backslash
        let basePath = fracturedImagePath.substring(0, lastSlashIndex ); // Get the base path up to and including the last backslash
        const lastSlashIndex1=basePath.lastIndexOf('\\');
        let basePath1 = basePath.substring(0, lastSlashIndex1 + 1);
        console.log(basePath1);
        fracturedImagePath3 = basePath1+ predictClass +'/'+  'prototype_' + prototypeIndex + '.jpg'; // Append the prototype file name

        const response = await fetch(`/image/${fracturedImagePath3}`);
        const image = await response.blob(); // Retrieve image data as a blob
        const selectedImage = document.getElementById('x-ray-image');
        selectedImage.src = URL.createObjectURL(image); // Set image data as src

    } catch (error) {
        console.error('Error fetching image URL:', error);
    }
}
async function displayImage(fracturedImagePath) {
    removeCanvas();
    let fracturedImagePath2;
    try {
        const clickedButton = event.currentTarget;
        // Remove the 'active' class from all buttons
        const allButtons = document.querySelectorAll('.sidebar button');
        allButtons.forEach(function (button) {
            button.classList.remove('active');
        });
        const commentSection = document.querySelector('.comment-section');
        // Add the 'active' class to the clicked button
        clickedButton.classList.add('active');
        fracturedImagePath2 = event.target.parentElement.querySelector('div[data-value]').getAttribute('data-value')
        const response = await fetch(`/image/${fracturedImagePath2}`);
        const image = await response.blob(); // Retrieve image data as a blob
        const selectedImage = document.getElementById('x-ray-image');
        selectedImage.src = URL.createObjectURL(image); // Set image data as src

    } catch (error) {
        console.error('Error fetching image URL:', error);
    }
}

document.addEventListener("DOMContentLoaded",  function save(accessionNumber) {
    const saveButton = document.getElementById("save-button");

    saveButton.addEventListener("click", function() {
        const accession_number = saveButton.getAttribute("accession_number");
        const commentInput = document.getElementById("comment-input").value;
        const selectedRating = document.querySelector("input[name='rating']:checked");

        if (selectedRating) {
            const ratingValue = selectedRating.value;

            // Assuming you have an endpoint to send data
            const endpoint = `/study/${accession_number}/save` // Replace with your actual endpoint

            // Prepare the data to send
            const commentAndRadpeerFormData = new FormData();
            if (commentInput) {
                commentAndRadpeerFormData.append('comment', commentInput);
            }
            commentAndRadpeerFormData.append('RADPEER_score', ratingValue);

            // Send the data using fetch
            fetch(endpoint, {
                method: "POST",
                body: commentAndRadpeerFormData
            })
                .then(response => response.json())
                .then(result => {
                    // Handle the response from the server if needed
                    console.log("Server response:", result);
                })
                .catch(error => {
                    console.error("Error:", error);
                });
            alert("RADPEER score and comment saved.")
        } else {
            alert("RADPEER score is mandatory.");
        }
    });
});
