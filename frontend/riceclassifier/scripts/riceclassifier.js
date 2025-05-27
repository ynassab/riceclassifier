"use strict"

const getDataAPIEndpoint = 'https://31rueu86eh.execute-api.us-east-1.amazonaws.com/';
let classifyLock = false;

const predictedClassIdToName = new Map();
predictedClassIdToName.set(0, 'Arborio');
predictedClassIdToName.set(1, 'Basmati');
predictedClassIdToName.set(2, 'Ipsala');
predictedClassIdToName.set(3, 'Jasmine');
predictedClassIdToName.set(4, 'Karacadag');

document.addEventListener('DOMContentLoaded', async () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const gallery = document.getElementById('gallery');
    const uploadedImage = document.getElementById('uploaded-image');
    const resultDiv = document.getElementById('result');
    const loadingDotContainer = document.getElementById('loading-dot-container');
    const inputPanel = document.getElementById('input-panel');

    resultDiv.textContent = '';  // Clear previous results

    const toBase64 = (file) =>
        new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);  // Returns full Base64 string (with prefix)
            reader.onerror = (error) => reject(error);
            reader.readAsDataURL(file);
        });

    async function handleFile(input) {
        try {
            if (classifyLock) return;
            classifyLock = true;

            let base64Image;

            if (input instanceof File) {  // Uploaded from file system
                if (!input || input.size === 0) {
                    console.log(input, input.size);
                    resultDiv.textContent = 'Uploaded file is empty or invalid.';
                    resultDiv.className = 'result error';
                    return;
                }
                if (!input.type.startsWith('image/')) {
                    resultDiv.textContent = `File must be an image. Instead, recieved: ${input.type}`;
                    resultDiv.className = 'result error';
                    return;
                };

                base64Image = await toBase64(input);

            } else if (typeof input === "string") {  // URL already in usable format
                base64Image = input;

            } else {
                resultDiv.textContent = `Unsupported input type. Expected file or string. Instead, received: ${typeof input}.`;
                resultDiv.className = 'result error';
                return;
            }

            resultDiv.textContent = '';
            loadingDotContainer.style.display = 'flex';
            inputPanel.classList.add('animate');

            updateImage(base64Image); // Display the image immediately
            let base64ImageNoPrefix = base64Image.split(',')[1];
            await sendImageToLambda(base64ImageNoPrefix);

        } catch (error) {
            console.error("Error processing the file:", error);

        } finally {
            loadingDotContainer.style.display = 'none';
            inputPanel.classList.remove('animate');
            classifyLock = false;
        }
    }

    function updateImage(base64Image) {
        uploadedImage.classList.remove('animate');
        uploadedImage.src = base64Image;

        setTimeout(() => {
            uploadedImage.classList.add('animate');
        }, 0);
    }

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('hover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('hover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('hover');

        const file = e.dataTransfer.files[0];

        if (file) {
            handleFile(file);
        } else {
            const src = e.dataTransfer.getData("text/plain"); // Check for preset image src
            if (src) {
                handleFile(src); // Handle preset image URL
            } else {
                alert("Error: Dropped content is not a valid image.");
            }
        }
    });

    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        handleFile(file);
    });

    gallery.addEventListener('dragstart', (e) => {
        if (e.target.tagName === 'IMG') {
            e.dataTransfer.setData('text/plain', e.target.src);
        }
    });

    uploadArea.addEventListener('drop', (e) => {
        const src = e.dataTransfer.getData('text/plain');
        if (src) {
            updateImage(src);
        }
    });

    async function sendImageToLambda(base64Image) {
        try {
            const response = await fetch(getDataAPIEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image })
            });

            const data = await response.json();
            if (response.ok) {
                let predictedClassName = predictedClassIdToName.get(data.predicted_class);
                typewriterEffect(resultDiv, `Predicted Class: ${predictedClassName}`, 15);
                resultDiv.className = 'result';
            } else {
                resultDiv.textContent = `Error: ${data.error || 'Unknown error'}`;
                resultDiv.className = 'result error';
            }
        } catch (error) {
            resultDiv.textContent = `Error: ${error.message}`;
            resultDiv.className = 'result error';
        }
    };

    function typewriterEffect(element, text, delayMilliseconds = 5) {
        let i = 0;
        let responseSoFar = '';

        function typeChunk() {
            if (i < text.length) {
                element.innerHTML += text[i];
                responseSoFar += text[i];
                i++;
                if (i % 25 === 0 || i === text.length) {
                    // Every n characters, re-render HTML to ensure tags and special characters appear properly
                    element.innerHTML = responseSoFar;
                }
                setTimeout(typeChunk, delayMilliseconds);
            } else {
                classifyLock = false;
            }
        }

        typeChunk();
    }
});
