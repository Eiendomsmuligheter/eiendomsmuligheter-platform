import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Container, Typography, Box, Button, CircularProgress } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

export const PaymentSuccess: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const location = useLocation();
  const navigate = useNavigate();
  
  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const sessionId = searchParams.get('session_id');

    if (sessionId) {
      fetch('/api/verify-payment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sessionId }),
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            setIsLoading(false);
          } else {
            setError('Kunne ikke verifisere betalingen');
            setIsLoading(false);
          }
        })
        .catch(err => {
          setError('Det oppstod en feil ved verifisering av betaling');
          setIsLoading(false);
        });
    }
  }, [location]);

  if (isLoading) {
    return (
      <Container maxWidth="sm" sx={{ textAlign: 'center', py: 8 }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Verifiserer betaling...
        </Typography>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="sm" sx={{ textAlign: 'center', py: 8 }}>
        <Typography variant="h5" color="error" gutterBottom>
          {error}
        </Typography>
        <Button
          variant="contained"
          color="primary"
          onClick={() => navigate('/payment')}
        >
          Prøv igjen
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="sm" sx={{ textAlign: 'center', py: 8 }}>
      <Box sx={{ mb: 4 }}>
        <CheckCircleIcon sx={{ fontSize: 64, color: 'success.main' }} />
      </Box>
      <Typography variant="h4" gutterBottom>
        Betaling Vellykket!
      </Typography>
      <Typography variant="body1" color="textSecondary" paragraph>
        Takk for at du valgte Eiendomsmuligheter! Du har nå tilgang til verdens beste eiendomsanalyse-plattform.
      </Typography>
      <Box sx={{ mt: 4 }}>
        <Button
          variant="contained"
          color="primary"
          size="large"
          onClick={() => navigate('/dashboard')}
        >
          Gå til Dashboard
        </Button>
      </Box>
    </Container>
  );
};