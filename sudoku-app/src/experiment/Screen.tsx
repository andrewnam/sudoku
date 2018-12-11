import React from 'react';
import ReactDOM from 'react-dom';

class Screen extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div className={"screen"}>
        Hello, World!
      </div>
    );
  }
}

export default Screen