#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
from hana_ml import dataframe
from hana_ml.dataframe import quotename
from pandas.testing import assert_frame_equal
import socket


class TestML_BaseTestClass(unittest.TestCase):
    tableDef = {}
    dfDef = {}

    def setUp(self):
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(__file__), 'config/e2edata.ini'))
        host_name = socket.gethostname()

        url = self.config.get('hana', 'url')
        port = int(self.config.get('hana', 'port'))
        pwd = self.config.get('hana', 'passwd')
        user = self.config.get('hana', 'user')
        self.conn = dataframe.ConnectionContext(url, port, user, pwd)
        self.conn.sql_tracer.enable_sql_trace(True)
        print("$"*50, host_name, url, port, user, pwd)

    def tearDown(self):
        self.conn.connection.close()

    def _createTable(self, tableName, select_statement=None):
        cursor = self.conn.connection.cursor()
        self._dropTableIgnoreError(tableName)

        if select_statement == None:
            statement = self.tableDef[tableName]
        else:
            statement = "CREATE COLUMN TABLE {} AS ({})".format(tableName, select_statement)

        try:
            cursor.execute(statement)
        finally:
            cursor.close()
            self.conn.connection.commit()

    def _dropTable(self, tableName):
        """ drop table """

        cursor = self.conn.connection.cursor()
        try:
            drop_sql = "DROP TABLE %s" % tableName
            cursor.execute(drop_sql)
        finally:
            cursor.close()
        self.conn.connection.commit()

    def _dropTableIgnoreError(self, tableName):
        try:
            self._dropTable(tableName)
        except:
            pass

    def _insertData(self, tableName, values):
        """ insert data into table """

        cursor = self.conn.connection.cursor()
        try:
            cursor.executemany("INSERT INTO " +
                               "{} VALUES ({})".format(tableName,
                               ', '.join(['?']*len(values[0]))), values)
            self.conn.connection.commit()
        finally:
            cursor.close()

    def _removeData(self, tableName):
        """delete data from table"""

        cursor = self.conn.connection.cursor()
        try:
            cursor.execute("DELETE FROM %s" % tableName)
            self.conn.connection.commit()
        finally:
            cursor.close()

    def _getData(self, tableName):
        """ verify data in table """

        cursor = self.conn.connection.cursor()
        try:
            cursor.execute("SELECT * FROM %s" % tableName)
            return cursor.fetchall()
        finally:
            cursor.close()

    def _assert_frame_not_equal(*args, **kwargs):
        try:
            assert_frame_equal(*args, **kwargs)
        except AssertionError:
            # frames are not equal
            pass
        else:
            # frames are equal
            raise AssertionError("DataFrames are equal when they shouldn't be.")
