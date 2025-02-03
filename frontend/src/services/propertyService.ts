import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api';

export interface PropertyAnalysis {
  property: {
    address: string;
    plotSize: number;
    buildYear: number;
    bra: number;
    coordinates: {
      lat: number;
      lng: number;
    };
  };
  regulations: {
    title: string;
    description: string;
    type: string;
    restrictions: string[];
  }[];
  potential: {
    options: {
      title: string;
      description: string;
      estimatedCost: number;
      potentialValue: number;
      requirements: string[];
    }[];
  };
  energyAnalysis: {
    rating: string;
    consumption: number;
    enovaSupport: {
      title: string;
      description: string;
      amount: number;
      requirements: string[];
    }[];
  };
  documents: {
    title: string;
    description: string;
    url: string;
    type: string;
  }[];
}

export const uploadProperty = async (file: File): Promise<{ fileId: string }> => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await axios.post(`${API_BASE_URL}/property/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error uploading property:', error);
    throw error;
  }
};

export interface AnalysisProgress {
  stage: string;
  progress: number;
  details?: string;
}

export interface AnalysisParams {
  address?: string;
  files?: File[];
  paymentIntentId?: string;
  onProgress?: (progress: AnalysisProgress) => void;
}

export const validateAddress = async (address: string): Promise<boolean> => {
  try {
    const response = await axios.post(`${API_BASE_URL}/property/validate-address`, {
      address,
    });
    return response.data.valid;
  } catch (error) {
    console.error('Error validating address:', error);
    throw new Error('Kunne ikke validere adressen');
  }
};

export const getPropertyDetails = async (address: string): Promise<{
  gnr: string;
  bnr: string;
  kommune: string;
  kommuneNr: string;
}> => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/property/details?address=${encodeURIComponent(address)}`
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching property details:', error);
    throw new Error('Kunne ikke hente eiendomsdetaljer');
  }
};

export const analyzeProperty = async ({
  address,
  files,
  paymentIntentId,
  onProgress,
}: AnalysisParams): Promise<PropertyAnalysis> => {
  try {
    const formData = new FormData();
    if (address) {
      formData.append('address', address);
    }
    if (files) {
      files.forEach((file, index) => {
        formData.append(\`files[\${index}]\`, file);
      });
    }
    if (paymentIntentId) {
      formData.append('paymentIntentId', paymentIntentId);
    }

    const response = await axios.post(`${API_BASE_URL}/property/analyze`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress({
            stage: 'upload',
            progress: percentCompleted,
            details: 'Laster opp filer...',
          });
        }
      },
    });

    // Starter analyseprosessen
    const analysisId = response.data.analysisId;
    let analysis = await pollAnalysisStatus(analysisId, onProgress);

    return analysis;
  } catch (error) {
    console.error('Error analyzing property:', error);
    throw error;
  }
};

const pollAnalysisStatus = async (
  analysisId: string,
  onProgress?: (progress: AnalysisProgress) => void
): Promise<PropertyAnalysis> => {
  const maxAttempts = 60; // 5 minutter med 5 sekunders intervall
  let attempts = 0;

  while (attempts < maxAttempts) {
    const status = await axios.get(
      `${API_BASE_URL}/property/analysis-status/${analysisId}`
    );

    if (status.data.completed) {
      return status.data.results;
    }

    if (onProgress) {
      onProgress({
        stage: status.data.stage,
        progress: status.data.progress,
        details: status.data.details,
      });
    }

    attempts++;
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }

  throw new Error('Analyse tok for lang tid');
};

export const getMunicipalityRegulations = async (
  municipalityCode: string
): Promise<any> => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/municipality/${municipalityCode}/regulations`
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching municipality regulations:', error);
    throw error;
  }
};

export const getPropertyHistory = async (propertyId: string): Promise<any> => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/property/${propertyId}/history`
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching property history:', error);
    throw error;
  }
};

export const generatePropertyDocuments = async (
  propertyId: string,
  documentType: string
): Promise<any> => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/property/${propertyId}/documents/generate`,
      { documentType }
    );
    return response.data;
  } catch (error) {
    console.error('Error generating property documents:', error);
    throw error;
  }
};

export const getEnovaSupportOptions = async (
  propertyId: string
): Promise<any> => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/property/${propertyId}/enova-support`
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching Enova support options:', error);
    throw error;
  }
};

export const saveAnalysisResults = async (
  propertyId: string,
  results: any
): Promise<any> => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/property/${propertyId}/analysis-results`,
      results
    );
    return response.data;
  } catch (error) {
    console.error('Error saving analysis results:', error);
    throw error;
  }
};