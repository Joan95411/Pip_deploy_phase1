
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
    const lastSlashIndex = fracturedImagePath.lastIndexOf('\\'); // Find the index of the last backslash
    let basePath = fracturedImagePath.substring(0, lastSlashIndex ); // Get the base path up to and including the last backslash
    const lastSlashIndex1=basePath.lastIndexOf('\\');
    let basePath1 = basePath.substring(0, lastSlashIndex1 + 1);

    await displayImage( fracturedImagePath);
    await displayPrototypes(prototypes_json,basePath1);

    const showOriginalButton = document.getElementById('show-original-button');
    const newButton = showOriginalButton.cloneNode(true);  // Clone the button to remove all event listeners
    showOriginalButton.parentNode.replaceChild(newButton, showOriginalButton);  // Replace old button with the new button
    // Add the new event listener with the updated fracturedImagePath
    newButton.addEventListener('click', function () {
        displayOriginalImage(fracturedImagePath);
    });
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
async function displayPrototypes(prototypes_json,basePath1) {

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
        let predictClass = prototype.predicted_class.toLowerCase();
        if (predictClass === 'not fractured') {
            predictClass = 'non_fractured';
        }
        // Add a click event listener to the index button
        indexButton.addEventListener('click', function () {
            displayPrototypeImage(prototype.prototype_index,basePath1,predictClass); // Call the function to display the image
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
    addPrototypeButton.addEventListener('click', function () {
        addNewPrototype(basePath1);
    });
    // Append the button below the table
    prototypesBar.appendChild(addPrototypeButton);

    // Create a new table for annotations
    const annotationTable = document.createElement('table');
    annotationTable.classList.add('annotation-table');
    const annotationHeader = document.createElement('h2');
    annotationHeader.classList.add('annotation-header');
    annotationHeader.textContent = `Annotations for Image`;
    prototypesBar.appendChild(annotationHeader);

    // Annotation Table Header
    const annotationHeaderRow = annotationTable.insertRow();
    const annotationCellID = annotationHeaderRow.insertCell();
    annotationCellID.textContent = 'Annotation ID';
    const annotationCellComment = annotationHeaderRow.insertCell();
    annotationCellComment.textContent = 'Comment';
    const basePath2 = basePath1.slice(0, -1);
    const lastSlashIndex = basePath2.lastIndexOf('/');
    let imageUid = basePath2.substring(lastSlashIndex + 1);
    const annotations = await fetchAnnotationsByImageUID(imageUid);

    // Populate the annotation table with fetched annotations
    annotations.forEach(function (annotation) {
        const annotationRow = annotationTable.insertRow();
        // Create a button for the annotation ID
        const annotationCellID = annotationRow.insertCell();
        const idButton = document.createElement('button');
        idButton.textContent = annotation.id;
        idButton.addEventListener('click', function () {
            displayAnnotatedImage(annotation.dir);  // Function to handle image display
        });
        annotationCellID.appendChild(idButton);
         // Create a cell for the comment
    const annotationCellComment = annotationRow.insertCell();
    const commentText = document.createElement('span');
    commentText.textContent = annotation.comment || 'No comment';

    // Create input field for editing (hidden by default)
    const commentInput = document.createElement('textarea');
    commentInput.value = annotation.comment;
    commentInput.style.display = 'none';  // Hide the textarea initially

    annotationCellComment.appendChild(commentText);
    annotationCellComment.appendChild(commentInput);

    // Create the "Edit Comment" button
    const annotationCellEdit = annotationRow.insertCell();
    const editButton = document.createElement('button');
    editButton.textContent = 'Edit Comment';

    // Event listener for editing and saving comment
    editButton.addEventListener('click', function () {
        if (editButton.textContent === 'Edit Comment') {
            // Start editing
            editButton.textContent = 'Finish Editing';
            commentText.style.display = 'none';  // Hide the text
            commentInput.style.display = 'inline';  // Show the input field
            commentInput.focus();
        } else {
            // Finish editing
            editButton.textContent = 'Edit Comment';
            commentText.textContent = commentInput.value || 'No comment';  // Update the display text
            commentText.style.display = 'inline';  // Show the text
            commentInput.style.display = 'none';  // Hide the input field

            // Call function to save the edited comment to the database
            saveAnnotationComment(annotation.id, commentInput.value);
        }
    });

    annotationCellEdit.appendChild(editButton);
    });

    // Append the annotation table to the prototypes bar
    prototypesBar.appendChild(annotationTable);

}

function saveAnnotationComment(annotationId, comment) {
    // Implement your API call here to save the comment
    // For example:
    fetch(`/edit-comment`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ id: annotationId, comment: comment })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Comment saved successfully');
        } else {
            console.error('Error saving comment:', data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

async function fetchAnnotationsByImageUID(imageUID) {
    try {
        const response = await fetch(`/fetch_annotations?image_uid=${imageUID}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const annotations = await response.json();
        return annotations;
    } catch (error) {
        console.error('Failed to fetch annotations:', error);
        return [];
    }
}

var clicks = [];
var canvas, context, imageObj;

function addNewPrototype(basePath1) {
    const addPrototypeButton = document.querySelector('.add-prototype-button');

    // Check if we're in the process of drawing
    if (addPrototypeButton.textContent === 'Finish Drawing') {
        // Save the drawing and reset the button
        saveDrawing3(basePath1);
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

function saveDrawing2(basePath1) {
    // Convert the canvas content (image + drawing) to a data URL
    const dataURL = canvas.toDataURL('image/png');
    basePath1 = basePath1.slice(0, -1);
    const lastSlashIndex = basePath1.lastIndexOf('/');
    let imageUid = basePath1.substring(lastSlashIndex + 1);
    const timestamp = new Date().toISOString().replace(/[:.-]/g, ''); // Replace colons, dots, and hyphens with empty strings
    const filename = `author_${timestamp}.png`;

    // Send the data URL, filename, and save directory to the server via a POST request
    fetch('/save-drawing2', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: dataURL,
            name: filename,
            image_uid:imageUid,
            directory: basePath1,
            points: clicks
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Drawing saved successfully on the server.');
        } else {
            console.error('Failed to save drawing on the server.');
        }
    })
    .catch(error => {
        console.error('Error while saving drawing:', error);
    });

    removeCanvas();
}

function saveDrawing3(basePath1) {
    let currentUserName = localStorage.getItem('selectedName')
    basePath1 = basePath1.slice(0, -1);
    const lastSlashIndex = basePath1.lastIndexOf('/');
    let imageUid = basePath1.substring(lastSlashIndex + 1);
    const timestamp = new Date().toISOString().replace(/[:.-]/g, ''); // Replace colons, dots, and hyphens with empty strings
    const filename = `${currentUserName}_${timestamp}.png`;
    console.log(currentUserName);
    const selectedImage = document.getElementById('x-ray-image');
    const sourceImageUrl = selectedImage.getAttribute('data-image-url');
    // Send the data URL, filename, and save directory to the server via a POST request
    fetch('/save-drawing3', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            name: filename,
            image_uid:imageUid,
            directory: basePath1,
            source_dir: sourceImageUrl,
            author: currentUserName,
            points: clicks
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Drawing saved successfully on the server.');
        } else {
            console.error('Failed to save drawing on the server.');
        }
    })
    .catch(error => {
        console.error('Error while saving drawing:', error);
    });

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

async function displayAnnotatedImage(imageUrl) {
    removeCanvas();
    try {
        const clickedButton = event.currentTarget;
        // Remove the 'active' class from all buttons
        const allButtons = document.querySelectorAll('.right-sidebar button');
        allButtons.forEach(function (button) {
            button.classList.remove('active');
        });
        // Add the 'active' class to the clicked button
        clickedButton.classList.add('active');

        const response = await fetch(`/image/${imageUrl}`);
        const image = await response.blob(); // Retrieve image data as a blob
        const selectedImage = document.getElementById('x-ray-image');
        selectedImage.src = URL.createObjectURL(image); // Set image data as src
        selectedImage.setAttribute('data-image-url', imageUrl);
    } catch (error) {
        console.error('Error fetching image URL:', error);
    }
}
// Function to display the prototype image
async function displayPrototypeImage(prototypeIndex,basePath1,predictClass) {

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

        fracturedImagePath3 = basePath1+ predictClass +'/'+  'prototype_' + prototypeIndex + '.jpg'; // Append the prototype file name

        const response = await fetch(`/image/${fracturedImagePath3}`);
        const image = await response.blob(); // Retrieve image data as a blob
        const selectedImage = document.getElementById('x-ray-image');
        selectedImage.src = URL.createObjectURL(image); // Set image data as src
        selectedImage.setAttribute('data-image-url', fracturedImagePath3);
    } catch (error) {
        console.error('Error fetching image URL:', error);
    }
}
async function displayOriginalImage(fracturedImagePath) {

    removeCanvas();
    let fracturedImagePath3;
    try {

        const lastSlashIndex = fracturedImagePath.lastIndexOf('\\');
        let imageName = fracturedImagePath.substring(lastSlashIndex + 1);
        imageName = imageName.replace('-fractured.jpg', '');
        let basePath = fracturedImagePath.substring(0, lastSlashIndex ); // Get the base path up to and including the last backslash
        const lastSlashIndex1=basePath.lastIndexOf('\\');
        let basePath1 = basePath.substring(0, lastSlashIndex1 + 1);
        fracturedImagePath3 = basePath1+  imageName + '.jpg'; // Append the prototype file name

        const response = await fetch(`/image/${fracturedImagePath3}`);
        const image = await response.blob(); // Retrieve image data as a blob
        const selectedImage = document.getElementById('x-ray-image');
        selectedImage.src = URL.createObjectURL(image); // Set image data as src
        selectedImage.setAttribute('data-image-url', fracturedImagePath3);
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
        clickedButton.classList.add('active');
        // fracturedImagePath2 = event.target.parentElement.querySelector('div[data-value]').getAttribute('data-value')
        const response = await fetch(`/image/${fracturedImagePath}`);
        const image = await response.blob(); // Retrieve image data as a blob
        const selectedImage = document.getElementById('x-ray-image');
        selectedImage.src = URL.createObjectURL(image); // Set image data as src
        selectedImage.setAttribute('data-image-url', fracturedImagePath);
    } catch (error) {
        console.error('Error fetching image URL:', error);
    }

}

function saveComment(accessionNumber) {
    const commentTextarea = document.getElementById(`comment-${accessionNumber}`);
    const comment = commentTextarea.value;

    fetch('/save-comment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ accession_number: accessionNumber, comment: comment })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Comment saved successfully');
        } else {
            console.error('Error saving comment:', data.error);
            alert('Failed to save comment.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while saving the comment.');
    });
}

document.addEventListener("DOMContentLoaded", function() {
    // Select the first button by its ID and trigger a click event
    const firstButton = document.querySelector('ul li button');
    if (firstButton) {
        firstButton.click();
    }
});

