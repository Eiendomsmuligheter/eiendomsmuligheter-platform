import React, { useState, useCallback } from 'react';
import { Box, Typography, Paper, Stepper, Step, StepLabel, Button } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useDropzone } from 'react-dropzone';
import { AddressSearch, FileUpload, PropertyDetails, AnalysisResults } from './components';
import { usePropertyAnalysis } from '../../hooks/usePropertyAnalysis';
import { PropertyData, AnalysisResult } from '../../types';

const AnalyzerContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  margin: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  maxWidth: 1200,
  marginLeft: 'auto',
  marginRight: 'auto',
}));

const steps = [
  'Velg analysemetode',
  'Last opp eller angi informasjon',
  'Gjennomfør analyse',
  'Se resultater'
];

const PropertyAnalyzer: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [analysisMethod, setAnalysisMethod] = useState<'address' | 'file' | 'link' | null>(null);
  const [propertyData, setPropertyData] = useState<PropertyData | null>(null);
  const { analyzeProperty, loading, error } = usePropertyAnalysis();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    // Håndter filopplasting
    const file = acceptedFiles[0];
    // TODO: Implementer filopplasting
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
      'application/pdf': ['.pdf']
    }
  });

  const handleAddressSubmit = async (address: string) => {
    try {
      const result = await analyzeProperty({ type: 'address', data: address });
      setPropertyData(result);
      nextStep();
    } catch (error) {
      console.error('Feil ved analyse av adresse:', error);
    }
  };

  const handleLinkSubmit = async (link: string) => {
    try {
      const result = await analyzeProperty({ type: 'link', data: link });
      setPropertyData(result);
      nextStep();
    } catch (error) {
      console.error('Feil ved analyse av link:', error);
    }
  };

  const nextStep = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const prevStep = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              Velg hvordan du vil analysere eiendommen
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
              <Button
                variant={analysisMethod === 'address' ? 'contained' : 'outlined'}
                onClick={() => setAnalysisMethod('address')}
              >
                Søk med adresse
              </Button>
              <Button
                variant={analysisMethod === 'file' ? 'contained' : 'outlined'}
                onClick={() => setAnalysisMethod('file')}
              >
                Last opp bilder/dokumenter
              </Button>
              <Button
                variant={analysisMethod === 'link' ? 'contained' : 'outlined'}
                onClick={() => setAnalysisMethod('link')}
              >
                Angi Finn.no lenke
              </Button>
            </Box>
          </Box>
        );

      case 1:
        return (
          <Box sx={{ mt: 4 }}>
            {analysisMethod === 'address' && (
              <AddressSearch onSubmit={handleAddressSubmit} />
            )}
            {analysisMethod === 'file' && (
              <FileUpload {...getRootProps()} inputProps={getInputProps()} />
            )}
            {analysisMethod === 'link' && (
              <Box>
                {/* TODO: Implementer link-input komponent */}
              </Box>
            )}
          </Box>
        );

      case 2:
        return (
          <PropertyDetails
            property={propertyData}
            onConfirm={nextStep}
            onEdit={prevStep}
          />
        );

      case 3:
        return (
          <AnalysisResults
            property={propertyData}
            onNewAnalysis={() => setActiveStep(0)}
          />
        );

      default:
        return null;
    }
  };

  return (
    <AnalyzerContainer>
      <Typography variant="h4" gutterBottom>
        Eiendomsanalyse
      </Typography>
      <Stepper activeStep={activeStep} alternativeLabel>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      {renderStepContent(activeStep)}
      <Box sx={{ mt: 4, display: 'flex', justifyContent: 'space-between' }}>
        {activeStep > 0 && (
          <Button onClick={prevStep} variant="outlined">
            Tilbake
          </Button>
        )}
        {activeStep < steps.length - 1 && analysisMethod && (
          <Button onClick={nextStep} variant="contained" disabled={loading}>
            Neste
          </Button>
        )}
      </Box>
      {error && (
        <Typography color="error" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}
    </AnalyzerContainer>
  );
};

export default PropertyAnalyzer;