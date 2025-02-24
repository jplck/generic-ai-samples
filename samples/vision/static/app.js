const fileUpload = document.getElementById("userPhoto");
const submitButton = document.getElementById("submit");
const vector = document.getElementById("vector");
submitButton.addEventListener('click', onSubmitButton);
const imageResults = document.getElementById("imageResults");

function onSubmitButton() {
    if (fileUpload.files.length == 0) {
        console.log("no file selected");
      }
      else {
        const file = fileUpload.files[0];
        console.log("posting file");
        const formData = new FormData();
        formData.append("file", file);
        const request = new XMLHttpRequest();
        const endpoint = `${window.location.protocol}//${window.location.host}/pictures`;
        request.open("POST", endpoint, true);
        request.onreadystatechange = () => {
          if (request.readyState === 4 && request.status === 200) {
            console.log(request.responseText);
            vector.innerHTML = request.responseText;
            document.body.lastVector = request.responseText;
            searchButton.disabled = false;
          }
          else if (request.readyState === 4) {
            console.log(request.status);
            vector.innerHTML = request.status;
          }
        };
        request.send(formData);
      }

}

const searchButton = document.getElementById("search");
searchButton.addEventListener('click', onSearchButton);

function onSearchButton() {
    const imageDetails = {
        vector: document.body.lastVector
    };

    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(imageDetails)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        imageResults.textContent = "List of images that are closest in order: " + data;
    })
    .catch(error => {
        console.error('Error:', error);
        imageResults.textContent = "Failed to create call"
    });
}