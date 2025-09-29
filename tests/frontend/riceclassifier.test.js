/**
 * @jest-environment jsdom
 */

let moduleExports;

function loadModuleWithDOM() {
    jest.resetModules();
    moduleExports = require('../../src/frontend/riceclassifier/scripts/riceclassifier.js');
}

describe('Rice Classifier Frontend', () => {
    let resultDiv, uploadedImage, loadingDotContainer, inputPanel;

    beforeEach(() => {
        document.body.innerHTML = `
            <div id="input-panel"></div>
            <div id="loading-dot-container"></div>
            <div id="result"></div>
            <img id="uploaded-image" />
        `;
        resultDiv = document.getElementById('result');
        uploadedImage = document.getElementById('uploaded-image');
        loadingDotContainer = document.getElementById('loading-dot-container');
        inputPanel = document.getElementById('input-panel');
        jest.useFakeTimers();

        loadModuleWithDOM();
    });

    afterEach(() => {
        jest.clearAllMocks();
        jest.clearAllTimers();
    });

    test('updateImage sets src and adds animate class', () => {
        moduleExports.updateImage('data:image/png;base64,abc123', uploadedImage);
        expect(uploadedImage.src).toContain('data:image/png;base64,abc123');
        jest.runAllTimers();
        expect(uploadedImage.classList.contains('animate')).toBe(true);
    });

    test('predictedClassIdToName maps correctly', () => {
        expect(moduleExports.predictedClassIdToName.get(0)).toBe('Arborio');
        expect(moduleExports.predictedClassIdToName.get(4)).toBe('Karacadag');
    });

    test('sendImageToLambda success response updates resultDiv', async () => {
        global.fetch = jest.fn(() =>
            Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ predicted_class: 1 })
            })
        );

        await moduleExports.sendImageToLambda('fakebase64', resultDiv);
        jest.runAllTimers();
        expect(resultDiv.textContent).toContain('Predicted Class: Basmati');
    });

    test('sendImageToLambda error response updates resultDiv', async () => {
        global.fetch = jest.fn(() =>
            Promise.resolve({
                ok: false,
                json: () => Promise.resolve({ error: 'Bad request' })
            })
        );

        await moduleExports.sendImageToLambda('fakebase64', resultDiv);
        expect(resultDiv.textContent).toContain('Error: Bad request');
    });

    test('typewriterEffect progressively updates element', () => {
        moduleExports.typewriterEffect(resultDiv, 'Test', 1);
        jest.advanceTimersByTime(10);
        expect(resultDiv.textContent).toContain('Test');
    });
});
