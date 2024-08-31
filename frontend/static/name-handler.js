document.addEventListener('DOMContentLoaded', function() {
    updateWelcomeMessage();
});

function showNameModal() {
    document.getElementById('nameModal').style.display = 'block';
    fetchNames();
    resetForm(); // Reset form fields when showing the modal
}

function closeModal() {
    document.getElementById('nameModal').style.display = 'none';
}

function fetchNames() {
    fetch('/get_names')
        .then(response => response.json())
        .then(data => {
            populateDropdown(data);
        })
        .catch(error => {
            console.error('Error fetching names:', error);
        });
}

function populateDropdown(names) {
    const select = document.getElementById('names');
    select.innerHTML = '<option value="">--Select--</option>'; // Reset options

    names.forEach(name => {
        const option = document.createElement('option');
        option.value = name.name;
        option.textContent = name.name;
        select.appendChild(option);
    });
}

document.getElementById('nameForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form from submitting the traditional way

    const formData = new FormData(this);
    const selectedName = formData.get('selectedName');
    const newName = formData.get('newName').trim();

    if (newName || selectedName) {
        const nameToUse = newName || selectedName;

        if (newName) {
            fetch('/add_name', {
                method: 'POST',
                body: new FormData(this)
            })
            .then(response => {
                if (response.ok) {
                    fetchNames();
                    localStorage.setItem('selectedName', nameToUse);
                    updateWelcomeMessage();
                    closeModal();
                } else {
                    console.error('Failed to add name');
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        } else {
            localStorage.setItem('selectedName', nameToUse);
            updateWelcomeMessage();
            closeModal();
        }
    }
});

function updateWelcomeMessage() {
    const name = localStorage.getItem('selectedName') || 'Guest';
    document.getElementById('userName').textContent = `Welcome, ${name}!`;
}

function resetForm() {
    const form = document.getElementById('nameForm');
    form.reset(); // This will clear all input fields and reset the select dropdown
}
