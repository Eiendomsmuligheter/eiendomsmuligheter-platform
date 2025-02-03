<<<<<<< HEAD
import React, { useState, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  TextField,
  CircularProgress
} from '@material-ui/core';
import { PropertyViewer } from './PropertyViewer';
import { AnalysisResults } from './AnalysisResults';
import { ModelControls } from './ModelControls';
import { uploadProperty, analyzeProperty } from '../services/propertyService';

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
    padding: theme.spacing(3),
  },
  paper: {
    padding: theme.spacing(2),
    textAlign: 'center',
    color: theme.palette.text.secondary,
  },
  uploadSection: {
    marginBottom: theme.spacing(3),
  },
  input: {
    display: 'none',
  },
}));

interface PropertyAnalyzerProps {
  onAnalysisComplete?: (results: any) => void;
}

export const PropertyAnalyzer: React.FC<PropertyAnalyzerProps> = ({ onAnalysisComplete }) => {
  const classes = useStyles();
  const [addressInput, setAddressInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleAddressSubmit = async () => {
    if (!addressInput.trim()) return;
    
    setIsAnalyzing(true);
    try {
      const results = await analyzeProperty({ address: addressInput });
      setAnalysisResults(results);
      if (onAnalysisComplete) onAnalysisComplete(results);
    } catch (error) {
      console.error('Error analyzing property:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileSubmit = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    try {
      const uploadResult = await uploadProperty(selectedFile);
      const results = await analyzeProperty({ fileId: uploadResult.fileId });
      setAnalysisResults(results);
      if (onAnalysisComplete) onAnalysisComplete(results);
    } catch (error) {
      console.error('Error processing file:', error);
    } finally {
      setIsAnalyzing(false);
=======
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
>>>>>>> 05b417208bb8af307dcc4b59d05bb20e32529392
    }
  };

  return (
<<<<<<< HEAD
    <Container className={classes.root}>
      <Grid container spacing={3}>
        <Grid item xs={12} className={classes.uploadSection}>
          <Paper className={classes.paper}>
            <Typography variant="h5" gutterBottom>
              Analyser din eiendom
            </Typography>
            <Grid container spacing={2} justifyContent="center">
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Skriv inn adresse"
                  variant="outlined"
                  value={addressInput}
                  onChange={(e) => setAddressInput(e.target.value)}
                />
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleAddressSubmit}
                  style={{ marginTop: '1rem' }}
                  disabled={isAnalyzing}
                >
                  Analyser adresse
                </Button>
              </Grid>
              <Grid item xs={12} md={6}>
                <input
                  accept="image/*,.pdf"
                  className={classes.input}
                  id="file-upload"
                  type="file"
                  onChange={handleFileUpload}
                />
                <label htmlFor="file-upload">
                  <Button
                    variant="contained"
                    component="span"
                    color="secondary"
                    disabled={isAnalyzing}
                  >
                    Last opp bilde/PDF
                  </Button>
                </label>
                {selectedFile && (
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={handleFileSubmit}
                    style={{ marginLeft: '1rem' }}
                    disabled={isAnalyzing}
                  >
                    Analyser fil
                  </Button>
                )}
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {isAnalyzing && (
          <Grid item xs={12}>
            <Paper className={classes.paper}>
              <CircularProgress />
              <Typography>Analyserer eiendom...</Typography>
            </Paper>
          </Grid>
        )}

        {analysisResults && (
          <>
            <Grid item xs={12} md={8}>
              <PropertyViewer property={analysisResults.property} />
            </Grid>
            <Grid item xs={12} md={4}>
              <ModelControls />
            </Grid>
            <Grid item xs={12}>
              <AnalysisResults results={analysisResults} />
            </Grid>
          </>
        )}
      </Grid>
    </Container>
=======
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
>>>>>>> 05b417208bb8af307dcc4b59d05bb20e32529392
  );
};

export default PropertyAnalyzer;