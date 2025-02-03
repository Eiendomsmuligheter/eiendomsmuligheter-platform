import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Button, Container, Typography, Paper, CircularProgress } from '@mui/material';
import { PropertyService } from '../services/PropertyService';
import { OmniverseViewer } from './OmniverseViewer';
import { AnalysisResults } from './AnalysisResults';

interface PropertyAnalyzerProps {
  onAnalysisComplete: (results: any) => void;
}

export const PropertyAnalyzer: React.FC<PropertyAnalyzerProps> = ({ onAnalysisComplete }) => {
  const [loading, setLoading] = useState(false);
  const [address, setAddress] = useState('');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setUploadedFiles(acceptedFiles);
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
      'application/pdf': ['.pdf']
    }
  });

  const handleAddressSubmit = async () => {
    setLoading(true);
    try {
      const results = await PropertyService.analyzeByAddress(address);
      setAnalysisResults(results);
      onAnalysisComplete(results);
    } catch (error) {
      console.error('Error analyzing property:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileAnalysis = async () => {
    if (!uploadedFiles.length) return;
    
    setLoading(true);
    try {
      const results = await PropertyService.analyzeFiles(uploadedFiles);
      setAnalysisResults(results);
      onAnalysisComplete(results);
    } catch (error) {
      console.error('Error analyzing files:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Paper elevation={3} sx={{ p: 4, my: 4 }}>
        <Typography variant="h4" gutterBottom>
          Eiendomsanalyse
        </Typography>

        {/* Address Input */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Analyser via adresse
          </Typography>
          <input
            type="text"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="Skriv inn adresse"
            style={{ width: '100%', padding: '10px', marginBottom: '10px' }}
          />
          <Button 
            variant="contained" 
            onClick={handleAddressSubmit}
            disabled={loading || !address}
          >
            Analyser Adresse
          </Button>
        </Box>

        {/* File Upload */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Last opp dokumenter
          </Typography>
          <div {...getRootProps()} style={{
            border: '2px dashed #ccc',
            padding: '20px',
            textAlign: 'center',
            cursor: 'pointer'
          }}>
            <input {...getInputProps()} />
            <Typography>
              Dra og slipp filer her, eller klikk for å velge filer
            </Typography>
          </div>
          {uploadedFiles.length > 0 && (
            <Button 
              variant="contained" 
              onClick={handleFileAnalysis}
              sx={{ mt: 2 }}
              disabled={loading}
            >
              Analyser Filer
            </Button>
          )}
        </Box>

        {/* Loading Indicator */}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {/* Results Display */}
        {analysisResults && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h5" gutterBottom>
              Analyseresultater
            </Typography>
            <OmniverseViewer modelData={analysisResults.modelData} />
            <AnalysisResults results={analysisResults} />
          </Box>
        )}
      </Paper>
    </Container>
  );
};