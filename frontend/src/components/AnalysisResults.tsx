import React, { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import {
  Paper,
  Typography,
  Tabs,
  Tab,
  Box,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Chip,
} from '@material-ui/core';
import {
  Home,
  Assessment,
  Description,
  AttachMoney,
  EmojiObjects,
  Gavel,
} from '@material-ui/icons';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && <Box p={3}>{children}</Box>}
    </div>
  );
};

const useStyles = makeStyles((theme) => ({
  root: {
    width: '100%',
  },
  header: {
    padding: theme.spacing(2),
    backgroundColor: theme.palette.primary.main,
    color: theme.palette.primary.contrastText,
  },
  tabs: {
    borderBottom: `1px solid ${theme.palette.divider}`,
  },
  tabContent: {
    marginTop: theme.spacing(2),
  },
  chip: {
    margin: theme.spacing(0.5),
  },
  statusSuccess: {
    color: theme.palette.success.main,
  },
  statusWarning: {
    color: theme.palette.warning.main,
  },
  actionButton: {
    marginTop: theme.spacing(2),
  },
  sectionTitle: {
    marginTop: theme.spacing(2),
    marginBottom: theme.spacing(1),
  },
}));

interface AnalysisResultsProps {
  results: {
    propertyInfo?: any;
    regulations?: any;
    potential?: any;
    energyAnalysis?: any;
    recommendations?: any;
    documents?: any;
  };
}

export const AnalysisResults: React.FC<AnalysisResultsProps> = ({ results }) => {
  const classes = useStyles();
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.ChangeEvent<{}>, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Paper className={classes.root}>
      <Box className={classes.header}>
        <Typography variant="h5">Analyserapport</Typography>
      </Box>

      <Tabs
        value={tabValue}
        onChange={handleTabChange}
        className={classes.tabs}
        variant="scrollable"
        scrollButtons="auto"
      >
        <Tab icon={<Home />} label="Eiendomsinfo" />
        <Tab icon={<Gavel />} label="Regelverk" />
        <Tab icon={<Assessment />} label="Potensial" />
        <Tab icon={<EmojiObjects />} label="Energi" />
        <Tab icon={<Description />} label="Dokumenter" />
      </Tabs>

      <TabPanel value={tabValue} index={0}>
        <Typography variant="h6" className={classes.sectionTitle}>
          Eiendomsinformasjon
        </Typography>
        <List>
          <ListItem>
            <ListItemText
              primary="Adresse"
              secondary={results.propertyInfo?.address}
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Tomtestørrelse"
              secondary={`${results.propertyInfo?.plotSize} m²`}
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Byggeår"
              secondary={results.propertyInfo?.buildYear}
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="BRA"
              secondary={`${results.propertyInfo?.bra} m²`}
            />
          </ListItem>
        </List>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <Typography variant="h6" className={classes.sectionTitle}>
          Gjeldende regulering
        </Typography>
        <List>
          {results.regulations?.map((regulation: any, index: number) => (
            <ListItem key={index}>
              <ListItemIcon>
                <Gavel />
              </ListItemIcon>
              <ListItemText
                primary={regulation.title}
                secondary={regulation.description}
              />
            </ListItem>
          ))}
        </List>
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <Typography variant="h6" className={classes.sectionTitle}>
          Utviklingspotensial
        </Typography>
        <Box>
          {results.potential?.options.map((option: any, index: number) => (
            <Paper key={index} style={{ padding: '1rem', marginBottom: '1rem' }}>
              <Typography variant="subtitle1">{option.title}</Typography>
              <Typography variant="body2">{option.description}</Typography>
              <Box mt={1}>
                <Chip
                  label={`Estimert kostnad: ${option.estimatedCost} NOK`}
                  className={classes.chip}
                />
                <Chip
                  label={`Potensiell verdiøkning: ${option.potentialValue} NOK`}
                  className={classes.chip}
                />
              </Box>
            </Paper>
          ))}
        </Box>
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        <Typography variant="h6" className={classes.sectionTitle}>
          Energianalyse
        </Typography>
        <List>
          <ListItem>
            <ListItemText
              primary="Energimerking"
              secondary={results.energyAnalysis?.rating}
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Årlig energiforbruk"
              secondary={`${results.energyAnalysis?.consumption} kWh/år`}
            />
          </ListItem>
        </List>
        <Typography variant="subtitle1" className={classes.sectionTitle}>
          Enova-støttemuligheter
        </Typography>
        <List>
          {results.energyAnalysis?.enovaSupport.map((support: any, index: number) => (
            <ListItem key={index}>
              <ListItemText
                primary={support.title}
                secondary={`Støttebeløp: ${support.amount} NOK`}
              />
            </ListItem>
          ))}
        </List>
      </TabPanel>

      <TabPanel value={tabValue} index={4}>
        <Typography variant="h6" className={classes.sectionTitle}>
          Tilgjengelige dokumenter
        </Typography>
        <List>
          {results.documents?.map((doc: any, index: number) => (
            <ListItem key={index} button>
              <ListItemIcon>
                <Description />
              </ListItemIcon>
              <ListItemText
                primary={doc.title}
                secondary={doc.description}
              />
              <Button
                variant="outlined"
                color="primary"
                size="small"
                onClick={() => window.open(doc.url, '_blank')}
              >
                Last ned
              </Button>
            </ListItem>
          ))}
        </List>
      </TabPanel>
    </Paper>
  );
};

export default AnalysisResults;