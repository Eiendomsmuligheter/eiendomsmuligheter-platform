import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Stepper,
  Step,
  StepLabel,
  CircularProgress,
  Alert,
  Grid,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SearchIcon from '@mui/icons-material/Search';
import { useHistory } from 'react-router-dom';
import { PropertyViewer } from './PropertyViewer';
import { AnalysisResults } from './AnalysisResults';
import { ModelControls } from './ModelControls';
import { analyzeProperty } from '../services/propertyService';

const DropzoneBox = styled(Box)(({ theme }) => ({
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(4),
  textAlign: 'center',
  cursor: 'pointer',
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
}));

const steps = [
  'Velg analysemetode',
  'Last opp filer eller angi adresse',
  'Utfør analyse',
  'Se resultater',
];

interface PropertyAnalyzerProps {
  onAnalysisComplete?: (results: any) => void;
}

const PropertyAnalyzer: React.FC<PropertyAnalyzerProps> = ({ onAnalysisComplete }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [address, setAddress] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const history = useHistory();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(acceptedFiles);
    setActiveStep(2);
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
      'application/pdf': ['.pdf'],
      'application/octet-stream': ['.dxf', '.dwg'],
    },
    maxFiles: 5,
  });

  const handleAddressSubmit = async () => {
    if (!address.trim()) {
      setError('Vennligst skriv inn en gyldig adresse');
      return;
    }
    setActiveStep(2);
    await performAnalysis();
  };

  const performAnalysis = async () => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      if (address) {
        formData.append('address', address);
      }
      files.forEach((file) => {
        formData.append('files', file);
      });

      const results = await analyzeProperty({ address, files });
      setAnalysisResults(results);
      if (onAnalysisComplete) onAnalysisComplete(results);
      
      setActiveStep(3);
      history.push('/results/' + results.modelUrl);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'En feil oppstod under analysen'
      );
      setActiveStep(0);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const renderContent = () => {
    switch (activeStep) {
      case 0:
        return (
          <Grid container spacing={3} justifyContent="center">
            <Grid item xs={12} md={6}>
              <Button
                fullWidth
                variant="contained"
                size="large"
                onClick={() => setActiveStep(1)}
              >
                Start analyse
              </Button>
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Søk på adresse
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    fullWidth
                    label="Skriv inn adresse"
                    value={address}
                    onChange={(e) => setAddress(e.target.value)}
                    placeholder="F.eks: Storgata 1, 0182 Oslo"
                  />
                  <Button
                    variant="contained"
                    startIcon={<SearchIcon />}
                    onClick={handleAddressSubmit}
                  >
                    Søk
                  </Button>
                </Box>
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Last opp dokumenter
                </Typography>
                <DropzoneBox {...getRootProps()}>
                  <input {...getInputProps()} />
                  <CloudUploadIcon sx={{ fontSize: 40, color: 'primary.main' }} />
                  <Typography variant="body1" sx={{ mt: 2 }}>
                    Dra og slipp filer her, eller klikk for å velge filer
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Støtter JPG, PNG, PDF, DXF og DWG
                  </Typography>
                </DropzoneBox>
              </Paper>
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Box textAlign="center">
            <CircularProgress />
            <Typography variant="h6" sx={{ mt: 2 }}>
              Analyserer eiendom...
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Dette kan ta noen minutter
            </Typography>
          </Box>
        );

      case 3:
        return (
          <Box>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <PropertyViewer
                  property={analysisResults.property}
                  modelUrl={analysisResults.modelUrl}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <ModelControls />
              </Grid>
              <Grid item xs={12}>
                <AnalysisResults results={analysisResults} />
              </Grid>
            </Grid>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      <Typography variant="h4" gutterBottom align="center">
        Eiendomsanalyse
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {renderContent()}
    </Box>
  );
};

export default PropertyAnalyzer;