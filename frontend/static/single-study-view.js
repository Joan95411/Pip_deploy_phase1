
async function backToAllStudies() {
    window.location.href = "/study/all";
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
    await displayPrototypes(prototypes_json);
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

async function displayPrototypes(prototypes_json) {

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
        dataCellID.textContent = prototype.prototype_index;
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
}
async function displayImage(fracturedImagePath) {
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
