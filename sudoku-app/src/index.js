import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

import Board from './board/Board.js'

// ========================================

ReactDOM.render(
  <Board dimX={2} dimY={2}/>,
  document.getElementById('root')
);
