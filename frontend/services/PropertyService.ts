import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

export class PropertyService {
  static async analyzeByAddress(address: string) {
    try {
      const response = await axios.post(`${API_BASE_URL}/property/analyze/address`, {
        address
      });
      return response.data;
    } catch (error) {
      console.error('Error in analyzeByAddress:', error);
      throw error;
    }
  }

  static async analyzeFiles(files: File[]) {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    try {
      const response = await axios.post(
        `${API_BASE_URL}/property/analyze/files`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error in analyzeFiles:', error);
      throw error;
    }
  }

  static async getPropertyDetails(propertyId: string) {
    try {
      const response = await axios.get(`${API_BASE_URL}/property/${propertyId}`);
      return response.data;
    } catch (error) {
      console.error('Error in getPropertyDetails:', error);
      throw error;
    }
  }

  static async generateDocuments(propertyId: string, documentTypes: string[]) {
    try {
      const response = await axios.post(`${API_BASE_URL}/property/${propertyId}/documents`, {
        documentTypes
      });
      return response.data;
    } catch (error) {
      console.error('Error in generateDocuments:', error);
      throw error;
    }
  }

  static async checkMunicipalityRegulations(propertyId: string) {
    try {
      const response = await axios.get(`${API_BASE_URL}/property/${propertyId}/regulations`);
      return response.data;
    } catch (error) {
      console.error('Error in checkMunicipalityRegulations:', error);
      throw error;
    }
  }

  static async getEnovaSupport(propertyId: string) {
    try {
      const response = await axios.get(`${API_BASE_URL}/property/${propertyId}/enova-support`);
      return response.data;
    } catch (error) {
      console.error('Error in getEnovaSupport:', error);
      throw error;
    }
  }

  static async analyze3DModel(propertyId: string) {
    try {
      const response = await axios.get(`${API_BASE_URL}/property/${propertyId}/3d-model`);
      return response.data;
    } catch (error) {
      console.error('Error in analyze3DModel:', error);
      throw error;
    }
  }
}