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
    }
  };

  return (
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
  );
};

export default PropertyAnalyzer;