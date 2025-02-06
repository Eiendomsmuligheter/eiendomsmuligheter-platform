import React, { useState } from 'react';
import {
  Container,
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  Link,
  CircularProgress,
  Alert,
  Stepper,
  Step,
  StepLabel
} from '@mui/material';
import { useAuth } from '../../hooks/useAuth';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { styled } from '@mui/material/styles';

// Stilede komponenter
const SignupContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  maxWidth: 600,
  margin: '0 auto',
  marginTop: theme.spacing(8),
}));

const Form = styled('form')(({ theme }) => ({
  width: '100%',
  marginTop: theme.spacing(1),
}));

interface SignupFormData {
  email: string;
  password: string;
  confirmPassword: string;
  name: string;
  company?: string;
  phone?: string;
}

const steps = ['Brukerinformasjon', 'Bedriftsdetaljer', 'Bekreftelse'];

export const Signup: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [formData, setFormData] = useState<SignupFormData>({
    email: '',
    password: '',
    confirmPassword: '',
    name: '',
    company: '',
    phone: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { signup } = useAuth();
  const navigate = useNavigate();
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };
  
  const validateStep = () => {
    switch (activeStep) {
      case 0:
        if (!formData.email || !formData.password || !formData.confirmPassword) {
          setError('Alle påkrevde felt må fylles ut');
          return false;
        }
        if (formData.password !== formData.confirmPassword) {
          setError('Passordene må være like');
          return false;
        }
        if (formData.password.length < 8) {
          setError('Passordet må være minst 8 tegn');
          return false;
        }
        break;
      case 1:
        if (!formData.name) {
          setError('Navn er påkrevd');
          return false;
        }
        break;
    }
    return true;
  };
  
  const handleNext = () => {
    if (validateStep()) {
      setError(null);
      setActiveStep((prev) => prev + 1);
    }
  };
  
  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
    setError(null);
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateStep()) return;
    
    try {
      setLoading(true);
      setError(null);
      
      await signup({
        email: formData.email,
        password: formData.password,
        name: formData.name,
        company: formData.company,
        phone: formData.phone,
      });
      
      navigate('/dashboard');
      
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'En feil oppstod under registrering'
      );
    } finally {
      setLoading(false);
    }
  };
  
  // Animasjonsvariabler
  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 }
  };
  
  const getStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <>
            <TextField
              variant="outlined"
              margin="normal"
              required
              fullWidth
              id="email"
              label="E-post"
              name="email"
              autoComplete="email"
              value={formData.email}
              onChange={handleChange}
              disabled={loading}
            />
            
            <TextField
              variant="outlined"
              margin="normal"
              required
              fullWidth
              name="password"
              label="Passord"
              type="password"
              id="password"
              value={formData.password}
              onChange={handleChange}
              disabled={loading}
            />
            
            <TextField
              variant="outlined"
              margin="normal"
              required
              fullWidth
              name="confirmPassword"
              label="Bekreft passord"
              type="password"
              id="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
              disabled={loading}
            />
          </>
        );
      case 1:
        return (
          <>
            <TextField
              variant="outlined"
              margin="normal"
              required
              fullWidth
              id="name"
              label="Fullt navn"
              name="name"
              value={formData.name}
              onChange={handleChange}
              disabled={loading}
            />
            
            <TextField
              variant="outlined"
              margin="normal"
              fullWidth
              id="company"
              label="Bedrift (valgfritt)"
              name="company"
              value={formData.company}
              onChange={handleChange}
              disabled={loading}
            />
            
            <TextField
              variant="outlined"
              margin="normal"
              fullWidth
              id="phone"
              label="Telefon (valgfritt)"
              name="phone"
              value={formData.phone}
              onChange={handleChange}
              disabled={loading}
            />
          </>
        );
      case 2:
        return (
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              Bekreft informasjon
            </Typography>
            
            <Typography variant="body1" gutterBottom>
              <strong>E-post:</strong> {formData.email}
            </Typography>
            
            <Typography variant="body1" gutterBottom>
              <strong>Navn:</strong> {formData.name}
            </Typography>
            
            {formData.company && (
              <Typography variant="body1" gutterBottom>
                <strong>Bedrift:</strong> {formData.company}
              </Typography>
            )}
            
            {formData.phone && (
              <Typography variant="body1" gutterBottom>
                <strong>Telefon:</strong> {formData.phone}
              </Typography>
            )}
          </Box>
        );
      default:
        return null;
    }
  };
  
  return (
    <Container component="main" maxWidth="sm">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        exit="exit"
        transition={{ duration: 0.3 }}
      >
        <SignupContainer elevation={3}>
          <Typography component="h1" variant="h5" gutterBottom>
            Registrer Ny Konto
          </Typography>
          
          <Stepper activeStep={activeStep} alternativeLabel sx={{ width: '100%', mb: 4 }}>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
          
          {error && (
            <Alert severity="error" sx={{ width: '100%', mb: 2 }}>
              {error}
            </Alert>
          )}
          
          <Form onSubmit={activeStep === steps.length - 1 ? handleSubmit : undefined}>
            {getStepContent(activeStep)}
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
              <Button
                onClick={handleBack}
                disabled={activeStep === 0 || loading}
              >
                Tilbake
              </Button>
              
              {activeStep === steps.length - 1 ? (
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  disabled={loading}
                >
                  {loading ? (
                    <CircularProgress size={24} color="inherit" />
                  ) : (
                    'Fullfør Registrering'
                  )}
                </Button>
              ) : (
                <Button
                  variant="contained"
                  onClick={handleNext}
                  disabled={loading}
                >
                  Neste
                </Button>
              )}
            </Box>
          </Form>
          
          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Link href="/login" variant="body2" underline="hover">
              Har du allerede en konto? Logg inn her
            </Link>
          </Box>
        </SignupContainer>
      </motion.div>
    </Container>
  );
};

export default Signup;