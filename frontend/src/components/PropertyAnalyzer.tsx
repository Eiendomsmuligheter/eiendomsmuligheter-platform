import React, { useState, useCallback, useEffect } from 'react';
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Chip,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SearchIcon from '@mui/icons-material/Search';
import PaymentIcon from '@mui/icons-material/Payment';
import DeleteIcon from '@mui/icons-material/Delete';
import { useHistory } from 'react-router-dom';
import { PropertyViewer } from './PropertyViewer';
import { AnalysisResults } from './AnalysisResults';
import { ModelControls } from './ModelControls';
import { analyzeProperty, validateAddress, getPropertyDetails } from '../services/propertyService';
import { loadStripe } from '@stripe/stripe-js';
import { Elements, PaymentElement, useStripe, useElements } from '@stripe/stripe-js';

// Stripe konfigurasjon
const stripePromise = loadStripe(process.env.REACT_APP_STRIPE_PUBLIC_KEY);

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

const PaymentDialog: React.FC<{
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  amount: number;
}> = ({ open, onClose, onSuccess, amount }) => {
  const stripe = useStripe();
  const elements = useElements();
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!stripe || !elements) return;

    setProcessing(true);
    const result = await stripe.confirmPayment({
      elements,
      confirmParams: {
        return_url: window.location.origin + '/payment-success',
      },
    });

    if (result.error) {
      setError(result.error.message || 'Betalingsfeil');
    } else {
      onSuccess();
    }
    setProcessing(false);
  };

  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>Betal for analyse</DialogTitle>
      <DialogContent>
        <Typography variant="body1" gutterBottom>
          Total kostnad: {amount} NOK
        </Typography>
        <form onSubmit={handleSubmit}>
          <PaymentElement />
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
          <DialogActions>
            <Button onClick={onClose}>Avbryt</Button>
            <Button
              type="submit"
              variant="contained"
              disabled={!stripe || processing}
              startIcon={<PaymentIcon />}
            >
              {processing ? 'Behandler...' : 'Betal'}
            </Button>
          </DialogActions>
        </form>
      </DialogContent>
    </Dialog>
  );
};

const PropertyAnalyzer: React.FC<PropertyAnalyzerProps> = ({ onAnalysisComplete }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [address, setAddress] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [showPayment, setShowPayment] = useState(false);
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const history = useHistory();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const validFiles = acceptedFiles.filter(file => {
      const maxSize = 50 * 1024 * 1024; // 50MB
      if (file.size > maxSize) {
        setError(`Filen ${file.name} er for stor. Maksimal størrelse er 50MB`);
        return false;
      }
      return true;
    });

    setFiles(prevFiles => [...prevFiles, ...validFiles]);
    if (validFiles.length > 0) {
      setError(null);
    }
  }, []);

  const removeFile = (index: number) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
      'application/pdf': ['.pdf'],
      'application/octet-stream': ['.dxf', '.dwg'],
      'application/vnd.dwg': ['.dwg'],
      'application/acad': ['.dwg'],
    },
    maxFiles: 10,
    multiple: true,
  });

  const handleAddressSubmit = async () => {
    if (!address.trim()) {
      setError('Vennligst skriv inn en gyldig adresse');
      return;
    }

    try {
      const isValid = await validateAddress(address);
      if (!isValid) {
        setError('Ugyldig adresse. Vennligst sjekk og prøv igjen.');
        return;
      }

      const details = await getPropertyDetails(address);
      setPropertyDetails(details);
      setShowPayment(true);

    } catch (err) {
      setError('Kunne ikke validere adressen. Vennligst prøv igjen.');
    }
  };

  const handlePaymentSuccess = async () => {
    setShowPayment(false);
    setActiveStep(2);
    await performAnalysis();
  };

  const performAnalysis = async () => {
    setIsAnalyzing(true);
    setError(null);
    let uploadProgressValue = 0;

    try {
      // Opprett betalingsintent først
      const paymentIntent = await createPaymentIntent({
        amount: calculateAnalysisCost(files.length),
        currency: 'nok',
        paymentMethodTypes: ['card'],
      });
      
      setClientSecret(paymentIntent.clientSecret);

      // Start analyseprosessen
      const formData = new FormData();
      if (address) {
        formData.append('address', address);
      }

      // Last opp filer med fremgangsmåling
      for (const file of files) {
        formData.append('files', file);
        uploadProgressValue += (100 / files.length);
        setUploadProgress(Math.min(95, uploadProgressValue)); // Max 95% until complete
      }

      const results = await analyzeProperty({ 
        address, 
        files,
        paymentIntentId: paymentIntent.paymentIntentId,
        onProgress: (progress) => {
          setUploadProgress(progress);
        }
      });

      setAnalysisResults(results);
      if (onAnalysisComplete) onAnalysisComplete(results);
      
      setActiveStep(3);
      setUploadProgress(100);
      history.push('/results/' + results.modelUrl);

    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'En feil oppstod under analysen'
      );
      setActiveStep(0);
    } finally {
      setIsAnalyzing(false);
      setUploadProgress(0);
    }
  };

  const renderContent = () => {
    switch (activeStep) {
      case 0:
        return (
          <Grid container spacing={3} justifyContent="center">
            <Grid item xs={12} md={8}>
              <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="h5" gutterBottom>
                  Velkommen til Eiendomsmuligheter
                </Typography>
                <Typography variant="body1" sx={{ mb: 3 }}>
                  Analyser din eiendom for å finne alle muligheter for utvikling og oppgradering
                </Typography>
                <Button
                  fullWidth
                  variant="contained"
                  size="large"
                  onClick={() => setActiveStep(1)}
                  sx={{ maxWidth: 400 }}
                >
                  Start analyse
                </Button>
              </Paper>
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
                <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                  <TextField
                    fullWidth
                    label="Skriv inn adresse"
                    value={address}
                    onChange={(e) => setAddress(e.target.value)}
                    placeholder="F.eks: Storgata 1, 0182 Oslo"
                    helperText="Vi søker automatisk i kommunens byggesaksarkiv"
                  />
                  <Button
                    variant="contained"
                    startIcon={<SearchIcon />}
                    onClick={handleAddressSubmit}
                  >
                    Søk
                  </Button>
                </Box>
                {propertyDetails && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle1">Eiendomsdetaljer:</Typography>
                    <Typography variant="body2">
                      GNR/BNR: {propertyDetails.gnr}/{propertyDetails.bnr}
                    </Typography>
                    <Typography variant="body2">
                      Kommune: {propertyDetails.kommune}
                    </Typography>
                  </Box>
                )}
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
                    Støtter JPG, PNG, PDF, DXF og DWG (Maks 50MB per fil)
                  </Typography>
                </DropzoneBox>

                {files.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Opplastede filer:
                    </Typography>
                    {files.map((file, index) => (
                      <Chip
                        key={index}
                        label={file.name}
                        onDelete={() => removeFile(index)}
                        sx={{ m: 0.5 }}
                      />
                    ))}
                  </Box>
                )}
              </Paper>
            </Grid>

            {(files.length > 0 || address) && (
              <Grid item xs={12}>
                <Button
                  variant="contained"
                  color="primary"
                  fullWidth
                  onClick={() => setShowPayment(true)}
                  startIcon={<PaymentIcon />}
                >
                  Gå videre til betaling
                </Button>
              </Grid>
            )}
          </Grid>
        );

      case 2:
        return (
          <Box textAlign="center">
            <CircularProgress />
            <Typography variant="h6" sx={{ mt: 2 }}>
              Analyserer eiendom...
            </Typography>
            <Box sx={{ width: '100%', mt: 2 }}>
              <LinearProgress variant="determinate" value={uploadProgress} />
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                {uploadProgress < 100
                  ? `Laster opp og analyserer... ${Math.round(uploadProgress)}%`
                  : 'Analyse fullført!'}
              </Typography>
            </Box>
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
              <Grid item xs={12}>
                <Button
                  variant="outlined"
                  onClick={() => {
                    setFiles([]);
                    setAddress('');
                    setActiveStep(0);
                  }}
                >
                  Start ny analyse
                </Button>
              </Grid>
            </Grid>
          </Box>
        );

      default:
        return null;
    }
  };

  const calculateAnalysisCost = (fileCount: number): number => {
    // Grunnpris
    let baseCost = 1499;
    
    // Tillegg per fil over 2 filer
    if (fileCount > 2) {
      baseCost += (fileCount - 2) * 299;
    }
    
    return baseCost;
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

      {showPayment && clientSecret && (
        <Elements stripe={stripePromise} options={{ clientSecret }}>
          <PaymentDialog
            open={showPayment}
            onClose={() => setShowPayment(false)}
            onSuccess={handlePaymentSuccess}
            amount={calculateAnalysisCost(files.length)}
          />
        </Elements>
      )}
    </Box>
  );
};

// Root eksport komponent som håndterer Stripe kontekst
const PropertyAnalyzerRoot: React.FC<PropertyAnalyzerProps> = (props) => {
  return (
    <Elements stripe={stripePromise}>
      <PropertyAnalyzer {...props} />
    </Elements>
  );
};

export default PropertyAnalyzerRoot;