import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Provider } from 'react-redux';
import store from './store';

// Components
import Navbar from './components/Navbar';
import PropertyAnalyzer from './components/PropertyAnalyzer';
import PropertyViewer from './components/PropertyViewer';
import AnalysisResults from './components/AnalysisResults';
import Dashboard from './components/Dashboard';
import PaymentPortal from './components/PaymentPortal';

// Theme configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
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
  },
});

function App() {
  return (
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <div className="App">
            <Navbar />
            <Switch>
              <Route exact path="/" component={Dashboard} />
              <Route path="/analyze" component={PropertyAnalyzer} />
              <Route path="/view/:propertyId" component={PropertyViewer} />
              <Route path="/results/:analysisId" component={AnalysisResults} />
              <Route path="/payment" component={PaymentPortal} />
            </Switch>
          </div>
        </Router>
      </ThemeProvider>
    </Provider>
  );
}

export default App;