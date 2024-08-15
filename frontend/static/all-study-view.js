
async function redirectToStudy(accessionNumber) {
    window.location.href = '/study/' + accessionNumber;
}

const pageInput = document.getElementById('page-input');
const prevButton = document.getElementById('prev-button');
const nextButton = document.getElementById('next-button');
const form = prevButton.closest('form'); // Find the form element

        // Handle the "Next" button click
        nextButton.addEventListener('click', function() {
            let currentPage = parseInt(pageInput.value, 10);
            pageInput.value = currentPage + 1;
            form.submit(); // Submit the form
        });

        // Handle the "Previous" button click
        prevButton.addEventListener('click', function() {
            let currentPage = parseInt(pageInput.value, 10);
            if (currentPage > 1) {
                pageInput.value = currentPage - 1;
                form.submit(); // Submit the form
            }
        });

        pageInput.addEventListener('change', function() {
            form.submit(); // Submit the form when page number is changed
        });
