import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

import Board from './board/Board.js'

// ========================================

ReactDOM.render(
  <Board dimX={3} dimY={3}/>,
  document.getElementById('root')
);
