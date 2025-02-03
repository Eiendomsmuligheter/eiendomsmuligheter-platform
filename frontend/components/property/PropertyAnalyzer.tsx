import React, { useState, useEffect } from 'react';
import { Box, Container, Typography, Button, TextField, CircularProgress } from '@mui/material';
import { Property3DViewer } from '../visualization/Property3DViewer';
import { AnalysisResults } from '../analysis/AnalysisResults';
import { FileUpload } from '../forms/FileUpload';
import { propertyService } from '../../services/propertyService';
import { useNvidiaOmniverse } from '../../hooks/useNvidiaOmniverse';
import styles from '../../styles/PropertyAnalyzer.module.css';

interface PropertyAnalyzerProps {
  onAnalysisComplete?: (results: any) => void;
}

export const PropertyAnalyzer: React.FC<PropertyAnalyzerProps> = ({ onAnalysisComplete }) => {
  // State management
  const [address, setAddress] = useState('');
  const [images, setImages] = useState<File[]>([]);
  const [listingUrl, setListingUrl] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Initialize Nvidia Omniverse
  const { initializeViewer, updateModel } = useNvidiaOmniverse();
  
  useEffect(() => {
    // Initialize 3D viewer when component mounts
    initializeViewer();
  }, []);
  
  const handleAddressChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setAddress(event.target.value);
    setError(null);
  };
  
  const handleUrlChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setListingUrl(event.target.value);
    setError(null);
  };
  
  const handleFileUpload = (files: File[]) => {
    setImages(files);
    setError(null);
  };
  
  const validateInput = (): boolean => {
    if (!address && !listingUrl && images.length === 0) {
      setError('Vennligst oppgi en adresse, URL eller last opp bilder');
      return false;
    }
    return true;
  };
  
  const handleAnalyze = async () => {
    if (!validateInput()) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      // Start analysis
      const results = await propertyService.analyzeProperty({
        address,
        images,
        listingUrl
      });
      
      // Update 3D model with analysis results
      await updateModel(results.buildingData);
      
      // Update state with results
      setAnalysisResults(results);
      
      // Notify parent component if callback provided
      if (onAnalysisComplete) {
        onAnalysisComplete(results);
      }
      
    } catch (err) {
      setError('Det oppstod en feil under analysen. Vennligst prøv igjen.');
      console.error('Analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  return (
    <Container maxWidth="lg" className={styles.container}>
      <Typography variant="h4" component="h1" gutterBottom>
        Eiendomsanalyse
      </Typography>
      
      <Box className={styles.inputSection}>
        <TextField
          fullWidth
          label="Adresse"
          variant="outlined"
          value={address}
          onChange={handleAddressChange}
          margin="normal"
        />
        
        <TextField
          fullWidth
          label="Finn.no URL eller annen boliglenke"
          variant="outlined"
          value={listingUrl}
          onChange={handleUrlChange}
          margin="normal"
        />
        
        <FileUpload
          onFilesSelected={handleFileUpload}
          acceptedTypes={['image/*', 'application/pdf']}
          maxFiles={10}
          maxSize={10000000} // 10MB
        />
        
        {error && (
          <Typography color="error" className={styles.error}>
            {error}
          </Typography>
        )}
        
        <Button
          variant="contained"
          color="primary"
          onClick={handleAnalyze}
          disabled={isAnalyzing}
          className={styles.analyzeButton}
        >
          {isAnalyzing ? (
            <>
              <CircularProgress size={24} />
              Analyserer...
            </>
          ) : (
            'Start Analyse'
          )}
        </Button>
      </Box>
      
      {analysisResults && (
        <Box className={styles.resultsSection}>
          <Property3DViewer
            buildingData={analysisResults.buildingData}
            className={styles.viewer}
          />
          
          <AnalysisResults
            results={analysisResults}
            className={styles.results}
          />
        </Box>
      )}
    </Container>
  );
};