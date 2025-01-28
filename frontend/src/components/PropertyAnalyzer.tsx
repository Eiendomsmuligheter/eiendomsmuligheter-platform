import React, { useState } from 'react';
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
  Alert
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PropertyViewer from './PropertyViewer';

const VisuallyHiddenInput = styled('input')`
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  height: 1px;
  overflow: hidden;
  position: absolute;
  bottom: 0;
  left: 0;
  white-space: nowrap;
  width: 1px;
`;

const steps = [
  'Last opp eller angi adresse',
  'Analyse pågår',
  'Se resultater og anbefalinger'
];

interface PropertyAnalyzerProps {
  onAnalysisComplete?: (result: any) => void;
}

const PropertyAnalyzer: React.FC<PropertyAnalyzerProps> = ({
  onAnalysisComplete
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [address, setAddress] = useState('');
  const [files, setFiles] = useState<FileList | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<any>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFiles(event.target.files);
      setError(null);
    }
  };

  const handleAddressChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setAddress(event.target.value);
    setError(null);
  };

  const validateInput = () => {
    if (!files && !address) {
      setError('Vennligst last opp filer eller angi en adresse');
      return false;
    }
    return true;
  };

  const startAnalysis = async () => {
    if (!validateInput()) return;

    try {
      setLoading(true);
      setActiveStep(1);

      const formData = new FormData();
      if (files) {
        Array.from(files).forEach(file => {
          formData.append('files', file);
        });
      }
      if (address) {
        formData.append('address', address);
      }

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Analyse feilet');
      }

      const result = await response.json();
      setAnalysisResult(result);
      onAnalysisComplete?.(result);
      setActiveStep(2);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'En feil oppstod under analysen');
      setActiveStep(0);
    } finally {
      setLoading(false);
    }
  };

  const renderStep = () => {
    switch (activeStep) {
      case 0:
        return (
          <Box sx={{ mt: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              fullWidth
              label="Eiendomsadresse"
              value={address}
              onChange={handleAddressChange}
              placeholder="F.eks: Storgata 1, 0182 Oslo"
            />

            <Typography variant="body1" sx={{ mt: 2 }}>
              Eller last opp filer:
            </Typography>

            <Button
              component="label"
              variant="contained"
              startIcon={<CloudUploadIcon />}
            >
              Last opp filer
              <VisuallyHiddenInput
                type="file"
                multiple
                onChange={handleFileChange}
                accept=".pdf,.png,.jpg,.jpeg,.dxf,.dwg"
              />
            </Button>

            {files && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="body2">
                  Valgte filer: {Array.from(files).map(f => f.name).join(', ')}
                </Typography>
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}

            <Button
              variant="contained"
              color="primary"
              onClick={startAnalysis}
              disabled={loading}
              sx={{ mt: 2 }}
            >
              Start analyse
            </Button>
          </Box>
        );

      case 1:
        return (
          <Box
            sx={{
              mt: 3,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 2
            }}
          >
            <CircularProgress />
            <Typography>Analyserer eiendom...</Typography>
          </Box>
        );

      case 2:
        return (
          <Box sx={{ mt: 3 }}>
            <PropertyViewer
              modelUrl={analysisResult?.modelUrl}
              initialData={analysisResult}
            />
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Eiendomsanalyse
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {renderStep()}
    </Paper>
  );
};

export default PropertyAnalyzer;