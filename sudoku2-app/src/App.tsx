import * as React from 'react';
import './App.css';

import 'primereact/resources/themes/nova-light/theme.css';
import 'primereact/resources/primereact.min.css';
import 'primeicons/primeicons.css';

import SudokuExperiment from "./app/SudokuExperiment";

class App extends React.Component {
  public render() {
    return (
      <SudokuExperiment/>
    );
  }
}

export default App;
