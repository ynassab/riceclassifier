/**
 * Rice Classifier Frontend
 *
 * Provides the frontend interface for the rice classifier web app. Handles user
 * image uploads (via drag-and-drop or file selection), communicates with the backend,
 * and displays classification results with animations.
 *
 * @author Yahia Nassab
 */

"use strict"

const getDataAPIEndpoint = 'https://31rueu86eh.execute-api.us-east-1.amazonaws.com/';
let lockClassification = false;

/**
 * Mapping of model output class indices to human-readable rice variety names.
 * These correspond to the 5 rice grain classes that the CNN model was trained to identify.
 */
const predictedClassIdToName = new Map();
predictedClassIdToName.set(0, 'Arborio');
predictedClassIdToName.set(1, 'Basmati');
predictedClassIdToName.set(2, 'Ipsala');
predictedClassIdToName.set(3, 'Jasmine');
predictedClassIdToName.set(4, 'Karacadag');

/**
 * Main application initialization and event listener setup.
 */
document.addEventListener('DOMContentLoaded', async () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const gallery = document.getElementById('gallery');
    const uploadedImage = document.getElementById('uploaded-image');
    const resultDiv = document.getElementById('result');
    const loadingDotContainer = document.getElementById('loading-dot-container');
    const inputPanel = document.getElementById('input-panel');

    resultDiv.textContent = '';  // Clear previous results

    /**
     * Convert a File object to a base64-encoded data URL string.
     *
     * @param {File} file - The File object to convert (typically from file input or drag-drop)
     * @returns {Promise<string>} Promise that resolves to a base64 data URL string
     *                            Format: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
     *
     */
    const toBase64 = (file) =>
        new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);  // Returns full Base64 string (with prefix)
            reader.onerror = (error) => reject(error);
            reader.readAsDataURL(file);
        });

    /**
     * Process and handle user image input for classification.
     *
     * This function serves as the main entry point for all image input methods
     * (file upload, drag-and-drop). It validates the input, converts it to the
     * appropriate format, updates the UI with loading states, and initiates
     * the classification process.
     *
     * @param {File|string} input - Either a File object from user upload or a string URL
     *                              from gallery drag-and-drop operations
     *
     */
    async function handleFile(input) {
        try {
            if (lockClassification) return;
            lockClassification = true;

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
            lockClassification = false;
        }
    }

    /**
     * Update the displayed image in the UI with smooth animation effects.
     *
     * @param {string} base64Image - Base64-encoded image data URL to display
     *                               Format: "data:image/jpeg;base64,..."
     *
     */
    function updateImage(base64Image) {
        uploadedImage.classList.remove('animate');
        uploadedImage.src = base64Image;

        setTimeout(() => {
            uploadedImage.classList.add('animate');
        }, 0);
    }

    // ===== Event Listeners for Drag and Drop Functionality =====

    gallery.addEventListener('dragstart', (e) => {
        if (e.target.tagName === 'IMG') {
            e.dataTransfer.setData('text/plain', e.target.src);
        }
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('hover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('hover');
    });

    /**
     * Processes both file drops and image URL drops from gallery.
     */
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

    // ===== Event Listeners for File Upload Interface =====

    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        handleFile(file);
    });

    /**
     * Send processed image data to AWS Lambda function for rice grain classification.
     *
     * This function handles the HTTP communication with the backend CNN model,
     * processes the response, and updates the UI with classification results.
     *
     * @param {string} base64Image - Base64-encoded image data (without data URL prefix)
     *                               Should be pure base64 string without "data:image/...;base64," prefix
     *
     */
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

    /**
     * Displays text with a typewriter animation effect.
     *
     * Characters are revealed progressively to create an engaging user experience
     * when displaying generated abstracts. HTML content is periodically re-rendered
     * to ensure proper display of tags and special characters.
     *
     * @param {HTMLElement} element - The DOM element to display the text in
     * @param {string} text - The text content to display with typewriter effect
     * @param {number} [delayMilliseconds=5] - Delay between each character in milliseconds
     *
     */
    function typewriterEffect(element, text, delayMilliseconds = 5) {
        let i = 0;
        let responseSoFar = '';

        /**
         * Internal recursive function that handles character-by-character text display.
         * Releases the generation lock when complete.
         */
        function typeChunk() {
            if (i < text.length) {
                element.innerHTML += text[i];
                responseSoFar += text[i];
                i++;
                // Periodically re-render HTML to ensure proper display
                if (i % 25 === 0 || i === text.length) {
                    element.innerHTML = responseSoFar;
                }
                setTimeout(typeChunk, delayMilliseconds);
            } else {
                // Release generation lock when animation completes
                lockGeneration = false;
            }
        }

        typeChunk();
    }
});
