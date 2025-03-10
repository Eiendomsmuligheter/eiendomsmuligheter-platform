import axios, { AxiosInstance } from 'axios';

// Typer
export interface Position3D {
  x: number;
  y: number;
  z: number;
}

export interface Dimensions3D {
  width: number;
  height: number;
  depth: number;
}

export interface BuildingModel {
  id?: string;
  name: string;
  position: Position3D;
  dimensions: Dimensions3D;
  color: string;
  type?: string;
  metadata?: Record<string, any>;
}

export interface TerrainModel {
  id?: string;
  width: number;
  depth: number;
  resolution: number;
  heightMap?: string;
  texture?: string;
}

export interface SceneData {
  buildings: BuildingModel[];
  terrain?: TerrainModel;
  cameraPosition?: Position3D;
  version?: string;
}

export interface ApiResponse {
  success: boolean;
  message?: string;
  [key: string]: any;
}

class VisualizationService {
  private apiClient: AxiosInstance;
  private static instance: VisualizationService;

  private constructor() {
    // Opprett Axios-klient med base URL
    this.apiClient = axios.create({
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api',
      timeout: 15000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Legg til interceptors for å håndtere tokens, osv.
    this.apiClient.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );
  }

  // Singleton pattern
  public static getInstance(): VisualizationService {
    if (!VisualizationService.instance) {
      VisualizationService.instance = new VisualizationService();
    }
    return VisualizationService.instance;
  }

  // API-metoder
  async createScene(sceneData: SceneData): Promise<ApiResponse> {
    try {
      const response = await this.apiClient.post('/visualization/scene', sceneData);
      return response.data;
    } catch (error) {
      console.error('Feil ved opprettelse av scene:', error);
      throw error;
    }
  }

  async getScene(sceneId: string): Promise<SceneData> {
    try {
      const response = await this.apiClient.get(`/visualization/scene/${sceneId}`);
      return response.data;
    } catch (error) {
      console.error('Feil ved henting av scene:', error);
      throw error;
    }
  }

  async updateScene(sceneId: string, sceneData: SceneData): Promise<ApiResponse> {
    try {
      const response = await this.apiClient.put(`/visualization/scene/${sceneId}`, sceneData);
      return response.data;
    } catch (error) {
      console.error('Feil ved oppdatering av scene:', error);
      throw error;
    }
  }

  async deleteScene(sceneId: string): Promise<ApiResponse> {
    try {
      const response = await this.apiClient.delete(`/visualization/scene/${sceneId}`);
      return response.data;
    } catch (error) {
      console.error('Feil ved sletting av scene:', error);
      throw error;
    }
  }

  async uploadHeightmap(file: File, width: number, depth: number, resolution: number): Promise<ApiResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('width', width.toString());
      formData.append('depth', depth.toString());
      formData.append('resolution', resolution.toString());

      const response = await this.apiClient.post('/visualization/terrain/heightmap', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Feil ved opplasting av høydekart:', error);
      throw error;
    }
  }

  async uploadTexture(file: File): Promise<ApiResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await this.apiClient.post('/visualization/terrain/texture', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Feil ved opplasting av tekstur:', error);
      throw error;
    }
  }
}

export default VisualizationService; 