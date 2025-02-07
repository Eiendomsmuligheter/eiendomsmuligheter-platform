import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import {
  Paper,
  Typography,
  Button,
  ButtonGroup,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@material-ui/core';
import {
  ZoomIn,
  ZoomOut,
  ThreeDRotation,
  ViewQuilt,
  Layers,
  Refresh,
} from '@material-ui/icons';

const useStyles = makeStyles((theme) => ({
  root: {
    padding: theme.spacing(2),
  },
  controlGroup: {
    marginBottom: theme.spacing(2),
  },
  buttonGroup: {
    marginBottom: theme.spacing(2),
    width: '100%',
  },
  button: {
    flexGrow: 1,
  },
  formControl: {
    width: '100%',
    marginBottom: theme.spacing(2),
  },
  slider: {
    width: '100%',
    marginTop: theme.spacing(2),
  },
}));

interface ModelControlsProps {
  onViewModeChange?: (mode: string) => void;
  onZoomChange?: (level: number) => void;
  onRotate?: (degrees: number) => void;
  onReset?: () => void;
}

export const ModelControls: React.FC<ModelControlsProps> = ({
  onViewModeChange,
  onZoomChange,
  onRotate,
  onReset,
}) => {
  const classes = useStyles();

  return (
    <Paper className={classes.root}>
      <Typography variant="h6" gutterBottom>
        Modellkontroller
      </Typography>

      {/* Visningsmoduser */}
      <div className={classes.controlGroup}>
        <FormControl className={classes.formControl}>
          <InputLabel>Visningsmodus</InputLabel>
          <Select
            onChange={(e) => onViewModeChange?.(e.target.value as string)}
            defaultValue="3d"
          >
            <MenuItem value="3d">3D-visning</MenuItem>
            <MenuItem value="floorplan">Plantegning</MenuItem>
            <MenuItem value="facade">Fasade</MenuItem>
            <MenuItem value="technical">Teknisk visning</MenuItem>
          </Select>
        </FormControl>
      </div>

      {/* Zoom-kontroller */}
      <div className={classes.controlGroup}>
        <Typography gutterBottom>Zoom</Typography>
        <ButtonGroup className={classes.buttonGroup}>
          <Button
            className={classes.button}
            onClick={() => onZoomChange?.(0.9)}
            startIcon={<ZoomOut />}
          >
            Zoom ut
          </Button>
          <Button
            className={classes.button}
            onClick={() => onZoomChange?.(1.1)}
            startIcon={<ZoomIn />}
          >
            Zoom inn
          </Button>
        </ButtonGroup>
      </div>

      {/* Rotasjonskontroller */}
      <div className={classes.controlGroup}>
        <Typography gutterBottom>Rotasjon</Typography>
        <Slider
          className={classes.slider}
          defaultValue={0}
          min={-180}
          max={180}
          onChange={(_, value) => onRotate?.(value as number)}
          valueLabelDisplay="auto"
          aria-labelledby="rotation-slider"
        />
      </div>

      {/* Lagvisning */}
      <div className={classes.controlGroup}>
        <Typography gutterBottom>Lag</Typography>
        <ButtonGroup className={classes.buttonGroup} orientation="vertical">
          <Button className={classes.button}>
            Arkitektur
          </Button>
          <Button className={classes.button}>
            Elektrisk
          </Button>
          <Button className={classes.button}>
            Rørlegging
          </Button>
          <Button className={classes.button}>
            Konstruksjon
          </Button>
        </ButtonGroup>
      </div>

      {/* Tilbakestill-knapp */}
      <Button
        variant="contained"
        color="secondary"
        fullWidth
        onClick={onReset}
        startIcon={<Refresh />}
      >
        Tilbakestill visning
      </Button>
    </Paper>
  );
};

export default ModelControls;