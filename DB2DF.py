from datetime import datetime
import pandas as pd
import math
import numpy


def db2uiDF(input):
    def calc_average_ui(filename):
        list = []
        ui = 0
        n_ui = 0
        date_aux = datetime.strptime("2004/1/1", "%Y/%m/%d")
        for i in filename.T.iteritems():
            new_date = datetime.strptime(
                str(int(i[1][0])) + "/" + str(int(i[1][1])) + "/" + str(int(i[1][2])),
                "%Y/%m/%d",
            )
            if new_date != date_aux:
                data = {"date": date_aux, "ui": None}
                date_aux = new_date

                if n_ui != 0:
                    data["ui"] = ui / n_ui
                ui = 0
                n_ui = 0
                list.append(data)

            if not numpy.isnan(i[1][6]):
                ui = ui + i[1][6]
                n_ui = n_ui + 1
        data = {"date": date_aux, "ui": None}
        if n_ui != 0:
            data["ui"] = ui / n_ui
        list.append(data)
        return pd.DataFrame(list)

    filename = pd.read_excel(input, "UI2")

    list = []
    ui = 0
    n_ui = 0
    df = calc_average_ui(filename)
    for i in df.T.iteritems():
        if i[1][0].weekday() == 5:
            data = {"date": i[1][0], "week": datetime.isocalendar(i[1][0])[1], "ui": ""}
            if n_ui != 0:
                data["ui"] = ui / n_ui
            ui = 0
            n_ui = 0
            list.append(data)
        if i[1][0].weekday() >= 0 and i[1][0].weekday() < 5:
            if not math.isnan(i[1][1]):
                ui = ui + i[1][1]
                n_ui = n_ui + 1
    data = {"date": i[1][0], "week": datetime.isocalendar(i[1][0])[1], "ui": ""}

    if n_ui != 0:
        data["ui"] = ui / n_ui
    list.append(data)
    df = pd.DataFrame(list)
    df.to_excel("v_ui.xlsx")
    return df


def db2chlorophyllDF(input):
    def calc_max_chlorophyll(filename):
        list = []
        max_chloroA = -1
        max_chloroB = -1
        max_chloroC = -1
        areas = [
            "L1",
            "L2",
            "L3",
            "L4",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "P0",
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "P9",
            "PA",
            "B1",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "EF",
        ]
        for p in areas:
            area_aux = "A0"
            date_aux = datetime.strptime("31/12/19", "%d/%m/%y")
            list_aux = []
            list_weeks = []
            for i in filename.T.iteritems():
                new_date = i[1][0]
                if (new_date != date_aux) or (area_aux != i[1][1].strip()):
                    if max_chloroA == -100:
                        max_chloroA = None
                    if max_chloroB == -100:
                        max_chloroB = None
                    if max_chloroC == -100:
                        max_chloroC = None
                    data = {
                        "data": date_aux,
                        "week": datetime.isocalendar(date_aux)[0]
                        + datetime.isocalendar(date_aux)[1] / 100,
                        "area": area_aux,
                        "chloroA": max_chloroA,
                        "chloroB": max_chloroB,
                        "chloroC": max_chloroC,
                    }
                    max_chloroA = -100
                    max_chloroB = -100
                    max_chloroC = -100
                    if area_aux == p:
                        if (
                            datetime.isocalendar(date_aux)[0]
                            + datetime.isocalendar(date_aux)[1] / 100
                            not in list_weeks
                        ):
                            list_aux.append(data)
                            list_weeks.append(
                                datetime.isocalendar(date_aux)[0]
                                + datetime.isocalendar(date_aux)[1] / 100
                            )
                    area_aux = i[1][1].strip()
                    date_aux = new_date

                if i[1][4] > max_chloroA and i[1][3] == "GFD":
                    max_chloroA = i[1][4]
                if i[1][5] > max_chloroB and i[1][3] == "GFD":
                    max_chloroB = i[1][5]
                if i[1][6] > max_chloroC and i[1][3] == "GFD":
                    max_chloroC = i[1][6]
            data = {
                "data": date_aux,
                "week": datetime.isocalendar(date_aux)[0]
                + datetime.isocalendar(date_aux)[1] / 100,
                "area": area_aux,
                "chloroA": max_chloroA,
                "chloroB": max_chloroB,
                "chloroC": max_chloroC,
            }
            if area_aux == p:
                if (
                    datetime.isocalendar(date_aux)[0]
                    + datetime.isocalendar(date_aux)[1] / 100
                    not in list_weeks
                ):
                    list_aux.append(data)
            if len(list) == 0:
                list = list_aux
                pdList = pd.DataFrame(list)
            else:
                pdAux = pd.DataFrame(list_aux)
                pdList = pdList.merge(pdAux, how="outer", on="week")

        output = pdList.iloc[::-1]
        return output

    filename = pd.read_excel(input)
    df = calc_max_chlorophyll(filename)
    df.to_excel("v_pig2.xlsx")
    return df


def db2nutrientsDF(input):
    def nutrients_x_area(filename):
        list = []
        week_aux = 0
        filename = pd.read_excel(filename)

        for i in filename.T.iteritems():
            new_date = i[1][0]
            area_aux = i[1][1].strip()
            data = {
                "date": new_date,
                "week": datetime.isocalendar(new_date)[1],
                "area": area_aux,
                "ammonium": i[1][2],
                "phosphate": i[1][3],
                "nitrate": i[1][4],
                "nitrite": i[1][5],
            }
            if datetime.isocalendar(new_date)[1] != week_aux:
                list.append(data)
            else:
                list[len(list) - 1]["ammonium"] = (
                    list[len(list) - 1]["ammonium"] + i[1][2]
                ) / 2
                list[len(list) - 1]["phosphate"] = (
                    list[len(list) - 1]["phosphate"] + i[1][3]
                ) / 2
                list[len(list) - 1]["nitrate"] = (
                    list[len(list) - 1]["nitrate"] + i[1][4]
                ) / 2
                list[len(list) - 1]["nitrite"] = (
                    list[len(list) - 1]["nitrite"] + i[1][5]
                ) / 2

            week_aux = datetime.isocalendar(new_date)[1]
        aux = 0
        for i in list:
            if i["week"] == aux:
                print(i["date"], i["area"])
            aux = i["week"]

        return pd.DataFrame(list)

    df = nutrients_x_area(input)
    df.to_excel("nuts.xlsx")
    return df


def db2dinophysisDF(input):
    def simplify_dinophysis(filename):
        filename = filename.iloc[::-1]
        list = []
        areas = [
            "L1",
            "L2",
            "L3",
            "L4",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "P0",
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "P9",
            "PA",
            "B1",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "EF",
        ]
        for p in areas:
            list_aux = []
            list_weeks = []
            for i in filename.T.iteritems():
                new_date = i[1][1]
                data = {
                    "date": new_date,
                    "week": datetime.isocalendar(new_date)[0]
                    + datetime.isocalendar(new_date)[1] / 100,
                    "area": i[1][0].strip(),
                    "dinoacum": i[1][4],
                    "dinoacut": i[1][5],
                    "dinocaud": i[1][6],
                    "dinospp": i[1][7],
                    "nitzs": i[1][8],
                    "alex": i[1][9],
                    "gymno": i[1][10],
                }
                if i[1][0].strip() == p and i[1][3] == 4:
                    if (
                        datetime.isocalendar(new_date)[0]
                        + datetime.isocalendar(new_date)[1] / 100
                        not in list_weeks
                    ):
                        list_aux.append(data)
                        list_weeks.append(
                            datetime.isocalendar(new_date)[0]
                            + datetime.isocalendar(new_date)[1] / 100
                        )
            if len(list) == 0:
                list = list_aux
                pdList = pd.DataFrame(list)
            else:
                pdAux = pd.DataFrame(list_aux)
                pdList = pdList.merge(pdAux, how="outer", on="week")
        output = pdList.iloc[::-1]
        return output

    filename = pd.read_excel(input)
    df = simplify_dinophysis(filename)
    df.to_excel("v_dino.xlsx")

    return df


def db2environmentalDF(input):
    def simplify_environmental_x_area(filename, pd_list):
        list = []
        temp = 0
        n_temp = 0
        o2 = 0
        n_o2 = 0
        sal = 0
        n_sal = 0
        termo0_4 = []
        termo4_8 = []
        termo8_12 = []
        alo0_4 = []
        alo4_8 = []
        alo8_12 = []
        areas_vistos = []
        weeks = []

        area_aux = filename.iloc[0, 0].strip()
        date_aux = datetime.strptime(filename.iloc[0, 3], "%d/%m/%Y %H:%M:%S")
        areas_vistos.append(area_aux)

        for i in filename.T.iteritems():
            new_date = datetime.strptime(i[1][3], "%d/%m/%Y %H:%M:%S")
            if new_date != date_aux or area_aux != i[1][0].strip():
                data = {
                    "date": date_aux,
                    "week": datetime.isocalendar(date_aux)[0]
                    + (datetime.isocalendar(date_aux)[1] / 100),
                    "area": area_aux,
                    "temperatura": None,
                    "termoclina": None,
                    "aloclina": None,
                    "oxigeno": None,
                    "salinidad": None,
                }

                if n_temp != 0:
                    data["temperatura"] = temp / n_temp
                temp = 0
                n_temp = 0

                if n_o2 != 0:
                    data["oxigeno"] = o2 / n_o2
                o2 = 0
                n_o2 = 0

                if n_sal != 0:
                    data["salinidad"] = sal / n_sal
                sal = 0
                n_sal = 0

                if len(termo4_8):
                    termo_1 = -9999
                    termo_2 = -9999
                    if len(termo0_4):
                        termo_1 = abs(
                            (sum(termo0_4) / len(termo0_4))
                            - (sum(termo4_8) / len(termo4_8))
                        )
                    if len(termo8_12):
                        termo_2 = abs(
                            (sum(termo4_8) / len(termo4_8))
                            - (sum(termo8_12) / len(termo8_12))
                        )
                    if termo_1 != -9999 or termo_2 != -9999:
                        if termo_1 < termo_2:
                            data["termoclina"] = termo_2
                        else:
                            data["termoclina"] = termo_1
                if len(alo4_8):
                    alo_1 = -9999
                    alo_2 = -9999
                    if len(alo0_4):
                        alo_1 = abs(
                            (sum(alo0_4) / len(alo0_4)) - (sum(alo4_8) / len(alo4_8))
                        )
                    if len(alo8_12):
                        alo_2 = abs(
                            (sum(alo4_8) / len(alo4_8)) - (sum(alo8_12) / len(alo8_12))
                        )
                    if termo_1 != -9999 or termo_2 != -9999:
                        if alo_1 < alo_2:
                            data["aloclina"] = alo_2
                        else:
                            data["aloclina"] = alo_1

                termo0_4 = []
                termo4_8 = []
                termo8_12 = []
                alo0_4 = []
                alo4_8 = []
                alo8_12 = []

                if (
                    datetime.isocalendar(date_aux)[0]
                    + (datetime.isocalendar(date_aux)[1] / 100)
                    not in weeks
                ):
                    list.append(data)
                    weeks.append(
                        datetime.isocalendar(date_aux)[0]
                        + (datetime.isocalendar(date_aux)[1] / 100)
                    )

                if i[1][0].strip() not in areas_vistos:
                    areas_vistos.append(i[1][0].strip())
                    if len(pd_list) == 0:
                        pd_list = pd.DataFrame(list)
                        list = []
                        weeks = []
                    else:
                        pdAux = pd.DataFrame(list)
                        pd_list = pd_list.merge(pdAux, how="outer", on="week")
                        list = []
                        weeks = []

                date_aux = new_date
                area_aux = i[1][0].strip()

            if i[1][4] != "-":
                if float(i[1][4].replace(",", ".")) > 0:
                    temp = temp + float(i[1][4].replace(",", "."))
                    n_temp = n_temp + 1
            if i[1][8] != "-":
                if float(i[1][8].replace(",", ".")) > 0:
                    o2 = o2 + float(i[1][8].replace(",", "."))
                    n_o2 = n_o2 + 1

            if i[1][10] != "-":
                if float(i[1][10].replace(",", ".")) < 12:
                    if i[1][6] != "-":
                        if float(i[1][6].replace(",", ".")) > 0:
                            sal = sal + float(i[1][6].replace(",", "."))
                            n_sal = n_sal + 1
                    if float(i[1][10].replace(",", ".")) < 8:
                        if float(i[1][10].replace(",", ".")) < 4:
                            if i[1][4] != "-":
                                if float(i[1][4].replace(",", ".")) > 0:
                                    termo0_4.append(float(i[1][4].replace(",", ".")))
                            if i[1][6] != "-":
                                if float(i[1][6].replace(",", ".")) > 0:
                                    alo0_4.append(float(i[1][6].replace(",", ".")))
                        else:
                            if i[1][4] != "-":
                                if float(i[1][4].replace(",", ".")) > 0:
                                    termo4_8.append(float(i[1][4].replace(",", ".")))
                            if i[1][6] != "-":
                                if float(i[1][6].replace(",", ".")) > 0:
                                    alo4_8.append(float(i[1][6].replace(",", ".")))
                    else:
                        if i[1][4] != "-":
                            if float(i[1][4].replace(",", ".")) > 0:
                                termo8_12.append(float(i[1][4].replace(",", ".")))
                        if i[1][6] != "-":
                            if float(i[1][6].replace(",", ".")) > 0:
                                alo8_12.append(float(i[1][6].replace(",", ".")))

        data = {
            "date": date_aux,
            "week": datetime.isocalendar(date_aux)[0]
            + (datetime.isocalendar(date_aux)[1] / 100),
            "area": area_aux,
            "temperatura": None,
            "termoclina": None,
            "aloclina": None,
            "oxigeno": None,
            "salinidad": None,
        }

        if n_temp != 0:
            data["temperatura"] = temp / n_temp

        if n_o2 != 0:
            data["oxigeno"] = o2 / n_o2

        if n_sal != 0:
            data["salinidad"] = sal / n_sal

        if len(termo4_8):
            termo_1 = -9999
            termo_2 = -9999
            if len(termo0_4):
                termo_1 = abs(
                    (sum(termo0_4) / len(termo0_4)) - (sum(termo4_8) / len(termo4_8))
                )
            if len(termo8_12):
                termo_2 = abs(
                    (sum(termo4_8) / len(termo4_8)) - (sum(termo8_12) / len(termo8_12))
                )
            if termo_1 != -9999 or termo_2 != -9999:
                if termo_1 < termo_2:
                    data["termoclina"] = termo_2
                else:
                    data["termoclina"] = termo_1
        if len(alo4_8):
            alo_1 = -9999
            alo_2 = -9999
            if len(alo0_4):
                alo_1 = abs((sum(alo0_4) / len(alo0_4)) - (sum(alo4_8) / len(alo4_8)))
            if len(alo8_12):
                alo_2 = abs((sum(alo4_8) / len(alo4_8)) - (sum(alo8_12) / len(alo8_12)))
            if termo_1 != -9999 or termo_2 != -9999:
                if alo_1 < alo_2:
                    data["aloclina"] = alo_2
                else:
                    data["aloclina"] = alo_1

        if (
            datetime.isocalendar(date_aux)[0]
            + (datetime.isocalendar(date_aux)[1] / 100)
            not in weeks
        ):
            list.append(data)

        if len(pd_list) == 0:
            pd_list = pd.DataFrame(list)
        else:
            pdAux = pd.DataFrame(list)
            pd_list = pd_list.merge(pdAux, how="outer", on="week")

        output = pd_list
        return output

    pd_list = []
    for i in input:
        filename = pd.read_excel(i)
        pd_list = simplify_environmental_x_area(filename, pd_list)

    pd_list.to_excel("v_amb_lorbe.xlsx")

    return pd_list


def db2sunDF(input):
    def simplify_data_sun(filename):
        listV1 = []
        filename = pd.read_excel(filename)
        for i in filename.T.iteritems():
            new_date = datetime.strptime(i[1][0], "%Y-%m-%d %H:%M:%S")
            data = {
                "date": new_date,
                "week": datetime.isocalendar(new_date)[1],
                "hSun": 0,
                "insolation": 0,
                "irradaytion": 0,
            }

            if i[1][1] != -9999:
                data["hSun"] = i[1][1]
            if i[1][2] != -9999:
                data["insolation"] = i[1][2]
            if i[1][3] != -9999:
                data["irradaytion"] = i[1][3]

            listV1.append(data)

        output = listV1

        return output

    def simplify_weeks_sun(filename):
        listV1 = []
        hSun = []
        insolation = []
        irradaytion = 0
        n_irradaytion = 0
        date_aux = datetime.strptime("1/3/14", "%d/%m/%y")
        for i in filename:
            new_date = i["date"]
            new_week = datetime.isocalendar(new_date)[1]
            week_aux = datetime.isocalendar(date_aux)[1]

            if new_week != week_aux:
                data = {
                    "date": date_aux,
                    "week": week_aux,
                    "hSun": 0,
                    "insolation": 0,
                    "irradaytion": 0,
                }
                date_aux = new_date

                if len(hSun) != 0:
                    data["hSun"] = sum(hSun) / len(hSun)
                if len(insolation) != 0:
                    data["insolation"] = sum(insolation) / len(insolation)
                if n_irradaytion != 0:
                    data["irradaytion"] = irradaytion / n_irradaytion

                irradaytion = 0
                n_irradaytion = 0
                hSun = []
                insolation = []

                listV1.append(data)

            if i["date"].weekday() >= 0 and i["date"].weekday() < 5:
                if not numpy.isnan(i["hSun"]):
                    hSun.append(i["hSun"])
                if not numpy.isnan(i["insolation"]):
                    insolation.append(i["insolation"])
                if not numpy.isnan(i["irradaytion"]):
                    irradaytion = irradaytion + i["irradaytion"]
                    n_irradaytion = n_irradaytion + 1

        data = {
            "date": date_aux,
            "week": datetime.isocalendar(date_aux)[1],
            "hSun": 0,
            "insolation": 0,
            "irradaytion": 0,
        }

        if len(hSun) != 0:
            data["hSun"] = sum(hSun) / len(hSun)
        if len(insolation) != 0:
            data["insolation"] = sum(insolation) / len(insolation)
        if n_irradaytion != 0:
            data["irradaytion"] = irradaytion / n_irradaytion

        listV1.append(data)

        output = pd.DataFrame(listV1)

        return output

    df = simplify_data_sun(input)
    df = simplify_weeks_sun(df)
    df.to_excel("v.xlsx")
    print(df)


def db2stateDF(input, day):
    def aux(input, day_aux):
        filename = pd.read_excel(input)
        name = input.split("_")
        year = name[2][0:4]

        areas = list(set(list(filename["_Zona"])))
        listPSP = []
        listASP = []
        listDSP = []

        for row in range(len(areas)):
            for i in filename.iteritems():
                x = i[0].split("/")

                if len(x) == 3:
                    mes = int(x[1]) + 1

                    if x[2][1] == "D":
                        day = int(x[2][2:4])

                    toxins = x[0][4:]

                if i[1][row] == "A" or i[1][row] == "C":
                    date_aux = datetime.strptime(
                        str(year) + "/" + str(mes) + "/" + str(day), "%Y/%m/%d"
                    )

                    if i[1][row] == "A":
                        state = 1
                    else:
                        state = 0

                    data = {
                        "date": date_aux,
                        "week": datetime.isocalendar(date_aux)[1],
                        "state": state,
                        "toxin": toxins,
                        "area": filename["_Zona"][row],
                        "estuary": filename["_Ria"][row],
                    }
                    if toxins == "PSP":
                        listPSP.append(data)
                    if toxins == "ASP":
                        listASP.append(data)
                    if toxins == "DSP":
                        listDSP.append(data)

        pdList = pd.DataFrame(listPSP)
        pdList = pd.merge(
            pdList,
            pd.DataFrame(listASP),
            how="left",
            left_on=["estuary", "area", "date"],
            right_on=["estuary", "area", "date"],
        )
        pdList = pd.merge(
            pdList,
            pd.DataFrame(listDSP),
            how="left",
            left_on=["estuary", "area", "date"],
            right_on=["estuary", "area", "date"],
        )

        pdOrdenada = pdList.sort_values(["estuary", "area", "date"])

        pdList = []
        list = []
        area_aux = pdOrdenada.T[0][2]
        for i in pdOrdenada.T.iteritems():
            data = {
                "date": i[1][1],
                "week": i[1][4],
                "PSP" + i[1][2]: i[1][0],
                "ASP" + i[1][2]: i[1][6],
                "DSP" + i[1][2]: i[1][9],
                "area": i[1][2],
                "estuary": i[1][3],
            }
            if area_aux != i[1][2]:
                if len(pdList) == 0:
                    pdList = pd.DataFrame(list)
                else:
                    pdList = pd.merge(
                        pdList,
                        pd.DataFrame(list),
                        how="left",
                        left_on=["date", "week"],
                        right_on=["date", "week"],
                    )
                list = []
                area_aux = i[1][2]

            if i[1][1].weekday() == day_aux:
                list.append(data)

        pdList = pd.merge(
            pdList,
            pd.DataFrame(list),
            how="left",
            left_on=["date", "week"],
            right_on=["date", "week"],
        )

        return pdList

    pdList = []

    if day == "lunes":
        valor_day = 0
    if day == "viernes":
        valor_day = 4
    for i in input:
        if len(pdList) == 0:
            pdList = aux(i, valor_day)
        else:
            list_aux = aux(i, valor_day)
            pdList = pdList.append(list_aux, ignore_index=True, sort=False)

    if day == "lunes":
        pdList.to_excel("v_states_lunes_final.xlsx")
    if day == "viernes":
        pdList.to_excel("v_states_viernes_final.xlsx")

    return pdList
