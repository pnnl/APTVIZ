"use strict";


(function() {

    window.addEventListener('load', init);

    function init() {
      const dropArea = qs(".drop-box");
      const button = dropArea.querySelector("button");
      // const dragText = dropArea.querySelector("header");
      const input = dropArea.querySelector("input");

      // let file;
      // var filename;

      button.onclick = () => {
        input.click();
      };

      input.addEventListener("change", function (e) {
        let fileNames = '';
        for (let i = 0; i < e.target.files.length; i++) {
          let fileName = e.target.files[i].name
          if (i != 0) {
            fileName = `, ${fileName}`
          };
          fileNames += fileName
        };

        qs("#upload-form h4").innerHTML = fileNames;
        qs('#upload-form p').classList.add('hidden');
        id('choose-file-btn').classList.add('hidden');
        id('upload-btn').classList.remove('hidden');

      });

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