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

export const analyzeProperty = async (params: {
  address?: string;
  fileId?: string;
}): Promise<PropertyAnalysis> => {
  try {
    const response = await axios.post(`${API_BASE_URL}/property/analyze`, params);
    return response.data;
  } catch (error) {
    console.error('Error analyzing property:', error);
    throw error;
  }
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