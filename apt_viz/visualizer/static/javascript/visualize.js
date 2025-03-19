"use strict";


(function() {

    window.addEventListener('load', init);

    function init() {
        id('manual').addEventListener('change', manualInputVis)
        id('louvain').addEventListener('change', manualInputVis)
    }

    function manualInputVis() {
      if (id('manual').checked) {
        id('n_clusters').style.visibility = 'visible'
      }
      else {
        id('n_clusters').style.visibility = 'hidden'
      };
      if (id('louvain').checked) {
        id('k_range').style.visibility = 'visible'
      }
      else {
        id('k_range').style.visibility = 'hidden'
      }
    }

// --------------------------- helper functions -------------------------- //
  /**
   * Helper function to return the response's result text if successful, otherwise
   * returns the rejected Promise result with an error status and corresponding text
   * @param {object} response - response to check for success/error
   * @return {object} - valid response if response was successful, otherwise rejected
   *                    Promise result
   */
  function checkStatus(response) {
    if (response.ok) {
      return response;
    } else {
      throw Error("Error in request: " + response.statusText);
    }
  }

  /**
   * Returns the element that has the ID attribute with the specified value.
   * @param {string} idName - element ID
   * @returns {object} DOM object associated with id.
   */
  function id(idName) {
    return document.getElementById(idName);
  }

  /**
   * Returns the first element that matches the given CSS selector.
   * @param {string} query - CSS query selector.
   * @returns {object} The first DOM object matching the query.
   */
  function qs(query) {
    return document.querySelector(query);
  }

  /**
   * Returns all elements that match the given CSS selector.
   * @param {string} query - CSS query selector.
   * @returns {array} an array of DOM objects matching the query.
   */
  function qsa(query) {
    return document.querySelectorAll(query);
  }
})();