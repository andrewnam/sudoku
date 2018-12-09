import React from 'react';
import PropTypes from 'prop-types';

import './Board.css';

class Cell extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      selected: false
    };
  }


  render() {
    if (this.props.board.candidatesView) {
      return (
        <div className="cell">

        </div>
      );
    } else {
      return (
        <button className={"cell-button"} onClick={() => this.props.board.cellOnClick(this)}>
          {this.props.board.state.digits[this.props.x][this.props.y]}
        </button>
      );
    }
  }
}

class Board extends React.Component {
  constructor(props) {
    super(props);
    this.maxDigit = this.props.dimX * this.props.dimY;
    this.state = {
      digits: Array(this.maxDigit).fill(Array(this.maxDigit).fill(null)),
      candidates: Array(this.maxDigit).fill(Array(this.maxDigit).fill(new Set())),
      candidatesView: false,
      selectedCell: null
    }
  }

  cellOnClick(cell) {
    this.setState((state, props) => {
      return {selectedCell: cell};
    });
    console.log(cell.props.x, cell.props.y)
  }

  renderCell(x, y) {
    return <Cell board={this}
                 x={x}
                 y={y}
                 key={(x,y)}/>
  }

  createGrid = () => {
    let grid = []

    for (let i = 0; i < this.maxDigit; i++) {
      let row = []
      for (let j = 0; j < this.maxDigit; j++) {
        row.push(this.renderCell(i, j))
      }
      grid.push(<div className={"row"} key={i}>{row}</div>)
    }
    return grid
  }

  render() {
    return (
      <div className={"grid"}>
        <h1>Hello, world!</h1>
        {this.createGrid()}
      </div>
    );
  }
}

Cell.propTypes = {
  board: PropTypes.instanceOf(Board).isRequired,
  x: PropTypes.number.isRequired,
  y: PropTypes.number.isRequired
}

export default Board
