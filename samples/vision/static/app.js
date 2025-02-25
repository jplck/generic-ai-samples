const fileUpload = document.getElementById("userPhoto");
const submitButton = document.getElementById("submit");
const vector = document.getElementById("vector");
submitButton.addEventListener('click', onSubmitButton);
const imageResults = document.getElementById("imageResults");
const visionButton = document.getElementById("vision");
visionButton.addEventListener('click', onVisionButton);

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


function onVisionButton() {
  const imageDetails = {
      picture1 : "images/2016-BMW-i3-94Ah-Protonic-Blue-33-kWh-Elektroauto-17.jpg",
      picture2 : "images/bmw-i3-ev_01.jpg"
  };

  fetch('/vision', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(imageDetails)
  })
  .then(response => response.json())
  .then(data => {
      console.log('Success:', data);
      visionResults.textContent = data;
  })
  .catch(error => {
      console.error('Error:', error);
      visionResults.textContent = "Failed to create call"
  });
}