import React from 'react';
import { Container, Typography, Box, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

export const PaymentCancel: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="sm" sx={{ textAlign: 'center', py: 8 }}>
      <Box sx={{ mb: 4 }}>
        <ErrorOutlineIcon sx={{ fontSize: 64, color: 'warning.main' }} />
      </Box>
      <Typography variant="h4" gutterBottom>
        Betaling Avbrutt
      </Typography>
      <Typography variant="body1" color="textSecondary" paragraph>
        Betalingen ble avbrutt. Ingen belastning har blitt gjort på din konto.
      </Typography>
      <Box sx={{ mt: 4 }}>
        <Button
          variant="contained"
          color="primary"
          size="large"
          onClick={() => navigate('/payment')}
        >
          Tilbake til Betalingsplaner
        </Button>
      </Box>
    </Container>
  );
};