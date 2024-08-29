#include <Arduino.h>
#include <math.h>

// Helper function to perform matrix multiplication
void matrixMultiply(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            float sum = 0;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
}

// Helper function to perform matrix transpose
void matrixTranspose(const float* A, float* AT, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            AT[j * rows + i] = A[i * cols + j];
        }
    }
}

// Helper function to compute the covariance matrix
void computeCovarianceMatrix(const float* data, float* covarianceMatrix, int rows, int cols) {
    float mean[cols] = {0};
    float* centeredData = new float[rows * cols];
    
    // Compute the mean of each column
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            mean[j] += data[i * cols + j];
        }
        mean[j] /= rows;
    }

    // Center the data
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            centeredData[i * cols + j] = data[i * cols + j] - mean[j];
        }
    }

    // Compute the covariance matrix
    float* centeredDataT = new float[cols * rows];
    matrixTranspose(centeredData, centeredDataT, rows, cols);
    matrixMultiply(centeredDataT, centeredData, covarianceMatrix, cols, rows, cols);

    delete[] centeredData;
    delete[] centeredDataT;
}

// PCA function to reduce dimensions
void performPCA(const float* data, float* reducedData, int rows, int cols, int reducedCols) {
    float* covarianceMatrix = new float[cols * cols];
    computeCovarianceMatrix(data, covarianceMatrix, rows, cols);

    // For simplicity, we assume the first 'reducedCols' columns of the covariance matrix are selected
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < reducedCols; ++j) {
            reducedData[i * reducedCols + j] = data[i * cols + j];
        }
    }

    delete[] covarianceMatrix;
}

void setup() {
    Serial.begin(115200);
    Serial.println("PCA Starts !!!");
    delay(1000);

    // Raw data: Replace with your data
    float rawData[] = {
        -0.0007, -0.0004, 100, 518.67, 641.82, 1589.7, 1400.6, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.3, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100, 39.06, 23.419,
        0.0019, -0.0003, 100, 518.67, 642.15, 1591.82, 1403.14, 14.62, 21.61, 553.75, 2388.04, 9044.07, 1.3, 47.49, 522.28, 2388.07, 8131.49, 8.4318, 0.03, 392, 2388, 100, 39, 23.4236,
        -0.0043, 0.0003, 100, 518.67, 642.35, 1587.99, 1404.2, 14.62, 21.61, 554.26, 2388.08, 9052.94, 1.3, 47.27, 522.42, 2388.03, 8133.23, 8.4178, 0.03, 390, 2388, 100, 38.95, 23.3442,
        0.0007, 0, 100, 518.67, 642.35, 1582.79, 1401.87, 14.62, 21.61, 554.45, 2388.11, 9049.48, 1.3, 47.13, 522.86, 2388.08, 8133.83, 8.3682, 0.03, 392, 2388, 100, 38.88, 23.3739,
        -0.0019, -0.0002, 100, 518.67, 642.37, 1582.85, 1406.22, 14.62, 21.61, 554, 2388.06, 9055.15, 1.3, 47.28, 522.19, 2388.04, 8133.8, 8.4294, 0.03, 393, 2388, 100, 38.9, 23.4044,
        -0.0043, -0.0001, 100, 518.67, 642.1, 1584.47, 1398.37, 14.62, 21.61, 554.67, 2388.02, 9049.68, 1.3, 47.16, 521.68, 2388.03, 8132.85, 8.4108, 0.03, 391, 2388, 100, 38.98, 23.3669,
        0.001, 0.0001, 100, 518.67, 642.48, 1592.32, 1397.77, 14.62, 21.61, 554.34, 2388.02, 9059.13, 1.3, 47.36, 522.32, 2388.03, 8132.32, 8.3974, 0.03, 392, 2388, 100, 39.1, 23.3774,
        -0.0034, 0.0003, 100, 518.67, 642.56, 1582.96, 1400.97, 14.62, 21.61, 553.85, 2388, 9040.8, 1.3, 47.24, 522.47, 2388.03, 8131.07, 8.4076, 0.03, 391, 2388, 100, 38.97, 23.3106,
        0.0008, 0.0001, 100, 518.67, 642.12, 1590.98, 1394.8, 14.62, 21.61, 553.69, 2388.05, 9046.46, 1.3, 47.29, 521.79, 2388.05, 8125.69, 8.3728, 0.03, 392, 2388, 100, 39.05, 23.4066
    };

    const int numRows = 9; // Number of rows in data
    const int numCols = 25; // Number of columns in data
    const int reducedCols = 5; // Number of dimensions to reduce to

    float reducedData[numRows * reducedCols];

    // Perform PCA
    performPCA(rawData, reducedData, numRows, numCols, reducedCols);

    // Print reduced data
    Serial.println("Reduced Data:");
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < reducedCols; ++j) {
            Serial.print(reducedData[i * reducedCols + j]);
            Serial.print(" ");
        }
        Serial.println();
    }

    Serial.println("PCA Done !!!");
}

void loop() {
    // Nothing to do here
}
