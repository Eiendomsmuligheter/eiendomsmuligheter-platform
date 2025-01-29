import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { PropertyAnalyzer, PropertyViewer, Dashboard, EnergyAnalysis } from './components';
import { Navigation, Footer } from './layout';
import { AuthProvider, PropertyProvider } from './contexts';

// Definerer et moderne og profesjonelt tema
const theme = createTheme({
  palette: {
    primary: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    secondary: {
      main: '#ff9800',
      light: '#ffb74d',
      dark: '#f57c00',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          padding: '8px 24px',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

function App() {
  return (
    <AuthProvider>
      <PropertyProvider>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Router>
            <div className="app">
              <Navigation />
              <main className="main-content">
                <Switch>
                  <Route exact path="/" component={Dashboard} />
                  <Route path="/analyze" component={PropertyAnalyzer} />
                  <Route path="/view/:propertyId" component={PropertyViewer} />
                  <Route path="/energy/:propertyId" component={EnergyAnalysis} />
                </Switch>
              </main>
              <Footer />
            </div>
          </Router>
        </ThemeProvider>
      </PropertyProvider>
    </AuthProvider>
  );
}

export default App;